import { pipeline } from '@xenova/transformers';
import dotenv from 'dotenv';
dotenv.config();
import axios from 'axios';
import express from 'express';
import * as cheerio from 'cheerio';
import fs from 'fs';
import path from 'path';
import { OpenAI } from 'openai';
import mongoose from 'mongoose';
import cors from 'cors';
import net from 'net';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
const pdfParse = require('pdf-parse');
import { franc } from 'franc';
import bcrypt from 'bcrypt';
import { OAuth2Client } from 'google-auth-library';

// 🆕 LangChain imports for web search agent
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { StateGraph } from "@langchain/langgraph";
import { Annotation } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import { MongoClient } from "mongodb";
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";

const app = express();
const PORT = 3000; // Express HTTP API
const TCP_PORT = 3001; // TCP Chatbot Server

// CORS and JSON middleware (already present, but ensure order)
app.use(express.json());
app.use(cors());

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

// MongoDB client for LangChain
let mongoClient = null;

// Initialize MongoDB client
async function initializeMongoClient() {
  if (!mongoClient) {
    mongoClient = new MongoClient(process.env.MONGODB_URI);
    await mongoClient.connect();
    console.log('MongoDB client for LangChain connected');
  }
  return mongoClient;
}

// 🆕 Web Search Agent using LangChain + Tavily
async function callWebSearchAgent(query, threadId = null, language = 'en') {
  try {
    const client = await initializeMongoClient();
    const dbName = "rag_web_agent";
    
    const GraphState = Annotation.Root({
      messages: Annotation({
        reducer: (x, y) => x.concat(y),
      }),
    });

    // Tavily Web Search Tool
    const tavily = new TavilySearchResults({
      apiKey: process.env.TAVILY_API_KEY,
      maxResults: 5,
    });

    const tools = [tavily];
    const toolNode = new ToolNode(tools);

    const model = new ChatOpenAI({
      modelName: "gpt-4o-mini",
      temperature: 0,
      apiKey: OPENAI_API_KEY,
    }).bindTools(tools);

    function shouldContinue(state) {
      const messages = state.messages;
      const lastMessage = messages[messages.length - 1];
      
      if (lastMessage.tool_calls?.length) {
        return "tools";
      }
      
      const content = typeof lastMessage.content === "string" ? lastMessage.content : "";
      if (content.toUpperCase().includes("FINAL ANSWER")) {
        return "__end__";
      }
      
      return "__end__";
    }

    async function callModel(state) {
      const systemMessage = language === 'es' 
        ? "Eres un asistente útil experto en las reglas y regulaciones de LAUSD que responde en español. Siempre utiliza la herramienta de búsqueda web para obtener información actualizada sobre LAUSD. Siempre provee enlaces a la información que encuentres."
        : "You are a helpful AI assistant expert in LAUSD rules and regulations. Always use the web search tool to get up-to-date information about LAUSD. Always provide links to the information you find.";

      const prompt = ChatPromptTemplate.fromMessages([
        [
          "system",
          `${systemMessage}
          
If you can fully answer the user's question, prefix your response with 'FINAL ANSWER:' and provide a comprehensive answer.
Do not repeat tasks endlessly. Always use the search tools when needed.
Available tools: {tool_names}.
Current time: {time}`,
        ],
        new MessagesPlaceholder("messages"),
      ]);

      const formattedPrompt = await prompt.formatMessages({
        time: new Date().toISOString(),
        tool_names: tools.map((tool) => tool.name).join(", "),
        messages: state.messages,
      });

      const result = await model.invoke(formattedPrompt);
      return { messages: [result] };
    }

    const workflow = new StateGraph(GraphState)
      .addNode("agent", callModel)
      .addNode("tools", toolNode)
      .addEdge("__start__", "agent")
      .addConditionalEdges("agent", shouldContinue)
      .addEdge("tools", "agent");

    const checkpointer = new MongoDBSaver({ client, dbName });
    const app = workflow.compile({ checkpointer });

    const config = {
      recursionLimit: 15,
      configurable: { 
        thread_id: threadId || `thread_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      }
    };

    const finalState = await app.invoke(
      {
        messages: [new HumanMessage(query)],
      },
      config
    );

    const finalMessage = finalState.messages[finalState.messages.length - 1].content;
    
    // Clean up the "FINAL ANSWER:" prefix if present
    const cleanedMessage = typeof finalMessage === 'string' 
      ? finalMessage.replace(/^FINAL ANSWER:\s*/i, '')
      : finalMessage;

    console.log('Web search agent response:', cleanedMessage);
    return cleanedMessage;
    
  } catch (error) {
    console.error("Web search agent error:", error);
    return language === 'es' 
      ? "Lamento que haya habido un error, ¿puedes intentar reformular tu pregunta? Intenta simplificar la pregunta. Gracias."
      : "I apologize for the error. Could you try rephrasing your question? Try to simplify the question. Thank you.";
  }
}

// --- TCP SERVER SETUP ---
const tcpServer = net.createServer((socket) => {
  socket.on('data', async (data) => {
    const str = data.toString();
    console.log('Received:', str);
    let parsed;
    try {
      parsed = JSON.parse(str);
    } catch (error) {
      // Not valid JSON, likely a health check or HTTP probe, just ignore
      console.warn('Non-JSON data received on TCP socket, ignoring.');
      return;
    }
    try {
      const { text, userId, targetLanguage } = parsed;
      console.log('Calling getRagAnswer...');
      const { answer, translated } = await getRagAnswer(text, targetLanguage, userId, 'tcp');
      console.log('Got answer:', answer);

      const responseData = JSON.stringify({
        response: answer,
        translated,
        timestamp: new Date().toISOString()
      });
      socket.write(responseData);
      socket.end();
    } catch (error) {
      console.error('Error processing TCP message:', error);
      socket.write(JSON.stringify({ error: 'Invalid request or server error.' }));
      socket.end();
    }
  });

  socket.on('error', (err) => {
    console.error('TCP socket error:', err);
  });
});

const TCP_HOST = '0.0.0.0';
tcpServer.listen(TCP_PORT, TCP_HOST, () => {
  console.log(`TCP Chatbot Server listening on ${TCP_HOST}:${TCP_PORT}`);
});
// --- END TCP SERVER SETUP ---

// Example: Minimal TCP echo server for testing with nc or telnet
const testTcpServer = net.createServer((socket) => {
  socket.on('data', (data) => {
    console.log('Received:', data.toString());
    socket.write('TCP ACK: ' + data);
  });
});

testTcpServer.listen(4000, () => {
  console.log('Raw TCP test server on port 4000');
});

// MongoDB connection
mongoose.connect(process.env.MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
  // ssl: true, // optional, Atlas uses SSL by default
  // tlsAllowInvalidCertificates: true, // REMOVED for production
});
mongoose.connection.on('connected', () => {
  console.log('MongoDB connected');
});
mongoose.connection.on('error', (err) => {
  console.error('MongoDB connection error:', err);
});

// MongoDB models
const messageSchema = new mongoose.Schema({
  role: { type: String, enum: ['user', 'assistant'], required: true },
  content: { type: String, required: true },
  timestamp: { type: Date, default: Date.now }
});
const conversationSchema = new mongoose.Schema({
  user: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true }, // Add this line
  title: String,
  messages: [messageSchema],
  createdAt: { type: Date, default: Date.now }
});
const Conversation = mongoose.model('Conversation', conversationSchema);
const documentChunkSchema = new mongoose.Schema({
  content: String,
  filename: String,
  embedding: [Number],
  metadata: {
    chunkIndex: Number,
    totalChunks: Number
  }
});
const DocumentChunk = mongoose.model('DocumentChunk', documentChunkSchema);

// New User model
const userSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true, trim: true },
  email:    { type: String, required: true, unique: true, trim: true },
  password: { type: String, required: true }, // Hashed password
  createdAt: { type: Date, default: Date.now }
});

// Hash password before saving
userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) return next();
  try {
    const salt = await bcrypt.genSalt(10);
    this.password = await bcrypt.hash(this.password, salt);
    next();
  } catch (err) {
    next(err);
  }
});

const User = mongoose.model('User', userSchema);

// Analytics schema for logging every /api/ask Q&A
const analyticsSchema = new mongoose.Schema({
  user: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true }, // Add this line
  question: String,
  answer: String,
  language: String,
  source: String,
  threadId: String,
  timestamp: { type: Date, default: Date.now }
});
const AnalyticsConversation = mongoose.model('AnalyticsConversation', analyticsSchema, 'analytics_conversations');

// --- EMBEDDING AND CHUNKING LOGIC (from server.js, keep as is for Render) ---
// Dummy chatbot response function (keeping original for TCP compatibility)
async function generateChatbotResponse(text, userId) {
  // Ensure PDF chunks are initialized
  const chunkCount = await DocumentChunk.countDocuments();
  if (chunkCount === 0) {
    await initializeDatabase();
  }

  // Find relevant chunks
  const chunks = await DocumentChunk.find({});
  if (chunks.length === 0) {
    return "No PDF documents found. Please ensure PDFs are in the correct directory.";
  }
  const chunkEmbeddings = chunks.map(c => c.embedding);
  const questionEmbedding = await getEmbedding(text);

  const similarities = chunkEmbeddings.map(e => cosineSimilarity(e, questionEmbedding));
  const topIndices = similarities
    .map((sim, idx) => ({ sim, idx }))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, 3)
    .map(obj => obj.idx);

  const ragContext = topIndices.map(idx =>
    `From ${chunks[idx].filename}:\n${chunks[idx].content}`
  ).join('\n\n---\n\n');

  const context = ragContext ? `Relevant LAUSD documents:\n\n${ragContext}` : '';

  if (!context) {
    return "Sorry, I couldn't find relevant information in the LAUSD documents.";
  }

  // Ask OpenAI
  const response = await makeOpenAIRequest({
    model: 'gpt-4o-mini',
    messages: [
      { role: 'system', content: 'You are a helpful assistant who answers questions using official LAUSD documents and be very specific about the answer, .' },
      { role: 'user', content: `${context}\n\nQuestion: ${text}` }
    ],
    temperature: 0.3
  });

  return response.data.choices?.[0]?.message?.content || 'No answer generated.';
}

// Dummy translation function
async function translateText(text, targetLanguage) {
  // Replace this with your actual translation logic if needed
  if (targetLanguage === 'es') {
    return `Traducción simulada: ${text}`;
  }
  return text;
}

// Utility: Chunk text
function chunkText(text, size = 1000) {
  const chunks = [];
  for (let i = 0; i < text.length; i += size) {
    chunks.push(text.slice(i, i + size));
  }
  return chunks;
}

// Utility: Cosine similarity
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Utility: Extract all PDFs from a directory
async function extractAllPdfs(pdfDir) {
  const absolutePdfDir = path.isAbsolute(pdfDir) ? pdfDir : path.join(process.cwd(), pdfDir);
  let results = [];
  try {
    const files = fs.readdirSync(absolutePdfDir).filter(f => f.endsWith('.pdf'));
    for (const file of files) {
      try {
        const dataBuffer = fs.readFileSync(path.join(absolutePdfDir, file));
        const pdfData = await pdfParse(dataBuffer);
        results.push({
          filename: file,
          text: pdfData.text
        });
        console.log(`Successfully extracted text from ${file}`);
      } catch (e) {
        console.error(`Error processing ${file}:`, e.message);
      }
    }
    return results;
  } catch (e) {
    console.error('Error in extractAllPdfs:', e.message);
    return [];
  }
}

// Embedding function (singleton)
let embedder = null;
async function getEmbedding(data) {
  if (!embedder) {
    embedder = await pipeline(
      'feature-extraction',
      'Xenova/nomic-embed-text-v1',
      {
        use_auth_token: process.env.HUGGINGFACE_API_KEY,
        execution_provider: 'cpu'
      }
    );
  }
  const results = await embedder(data, { pooling: 'mean', normalize: true });
  return Array.from(results.data);
}

// OpenAI request with retry
async function makeOpenAIRequest(data) {
  const maxRetries = 3;
  let retryCount = 0;
  while (retryCount < maxRetries) {
    try {
      const response = await axios.post(
        'https://api.openai.com/v1/chat/completions',
        data,
        {
          headers: {
            'Authorization': `Bearer ${OPENAI_API_KEY}`,
            'Content-Type': 'application/json'
          },
          timeout: 10000
        }
      );
      return response;
    } catch (error) {
      console.error(`Request failed (attempt ${retryCount + 1}):`, error.message);
      retryCount++;
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  throw new Error(`Failed to get response after ${maxRetries} retries.`);
}

// Initialize database with PDF chunks (no duplicates)
async function initializeDatabase() {
  try {
    const pdfs = await extractAllPdfs('./pdfs');
    const allChunks = [];
    pdfs.forEach(pdf => {
      const chunks = chunkText(pdf.text);
      chunks.forEach((chunk, idx) => allChunks.push({
        content: chunk,
        filename: pdf.filename,
        chunkIndex: idx,
        totalChunks: chunks.length
      }));
    });

    let storedCount = 0;
    for (const chunk of allChunks) {
      const existingChunk = await DocumentChunk.findOne({
        content: chunk.content,
        filename: chunk.filename,
        'metadata.chunkIndex': chunk.chunkIndex
      });
      if (!existingChunk) {
        try {
          const embedding = await getEmbedding(chunk.content);
          const chunkDoc = new DocumentChunk({
            content: chunk.content,
            filename: chunk.filename,
            embedding: embedding,
            metadata: {
              chunkIndex: chunk.chunkIndex,
              totalChunks: chunk.totalChunks
            }
          });
          await chunkDoc.save();
          storedCount++;
        } catch (embeddingError) {
          console.error('Error generating embedding or saving chunk:', chunk.filename, chunk.chunkIndex, embeddingError);
        }
      } else {
        console.log('Skipping duplicate chunk:', chunk.filename, chunk.content.slice(0, 20) + '...');
      }
    }
    console.log(`Database initialized. Stored ${storedCount} new PDF chunks.`);
  } catch (error) {
    console.error('Error initializing database:', error);
  }
}

// Conversation endpoints
app.post('/api/conversations', async (req, res) => {
  try {
    const { title, userId } = req.body;
    const conversation = new Conversation({ user: userId, title: title || 'Untitled', messages: [] });
    await conversation.save();
    res.status(201).json(conversation);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

// Example for conversations
app.get('/api/conversations', async (req, res) => {
  const { userId } = req.query; // Or from auth/session
  try {
    const conversations = await Conversation.find({ user: userId }).sort({ createdAt: -1 });
    res.json(conversations);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/conversations/:id', async (req, res) => {
  try {
    const conversation = await Conversation.findById(req.params.id);
    if (!conversation) return res.status(404).json({ error: 'Not found' });
    res.json(conversation);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/api/conversations/:id/messages', async (req, res) => {
  try {
    const { role, content } = req.body;
    const conversation = await Conversation.findById(req.params.id);
    if (!conversation) return res.status(404).json({ error: 'Not found' });
    conversation.messages.push({ role, content });
    await conversation.save();
    res.json(conversation);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

app.patch('/api/conversations/:id', async (req, res) => {
  try {
    const conversation = await Conversation.findByIdAndUpdate(req.params.id, req.body, { new: true });
    if (!conversation) return res.status(404).json({ error: 'Not found' });
    res.json(conversation);
  } catch (err) {
    res.status(400).json({ error: err.message });
  }
});

app.delete('/api/conversations/:id', async (req, res) => {
  try {
    await Conversation.findByIdAndDelete(req.params.id);
    res.json({ message: 'Conversation deleted' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// 🆕 New endpoint for web search agent
app.post('/api/web-search', async (req, res) => {
  const { question, language = 'en', threadId } = req.body;
  
  if (!question) {
    return res.status(400).json({ error: 'Question is required.' });
  }

  try {
    const answer = await callWebSearchAgent(question, threadId, language);
    
    return res.json({
      question,
      answer,
      language,
      threadId: threadId || 'auto-generated',
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    console.error('Error in /api/web-search:', err);
    return res.status(500).json({ 
      error: 'Failed to process web search question.', 
      details: err.message 
    });
  }
});


// Updated main RAG endpoint with automatic web search fallback
app.post('/api/ask', async (req, res) => {
  const { question, threadId, userId } = req.body;
  if (!question) {
    return res.status(400).json({ error: 'Question is required.' });
  }

  // Detect language
  let detectedLang = franc(question, { minLength: 3 });
  let language = 'en'; // default
  if (detectedLang === 'spa') language = 'es';
  else if (detectedLang === 'eng') language = 'en';

  try {
    // First try RAG system
    const { answer: ragAnswer, translated } = await getRagAnswer(question, language, null, 'http');

    // Initialize needsWebSearch variable
    let needsWebSearch = false;

    // Check the appropriate answer based on language
    const answerToCheck = language === 'es' ? translated : ragAnswer;

    // Special case: If answer starts with "Traducción simulada:" it's definitely a fallback
    if (typeof answerToCheck === 'string' && answerToCheck.startsWith("Traducción simulada:")) {
      needsWebSearch = true;
    }
    
    // Check if answer contains fallback phrases
    const fallbackPhrases = [
      // English phrases
      "couldn't find", "no relevant", "not found", "not mentioned", 
      "I recommend", "you should check", "please contact", "I suggest",
      "I'm not sure", "I don't know", "unable to find", "doesn't specify",
      "doesn't mention", "not addressed", "not covered", "not available",
      "not listed", "not provided", "not discussed", "refer to",
      "check the website", "contact the", "look up", "search for",
      "verify with", "consult the", "no information", "no details",
      "not in the documents", "not in our records", "not currently available",
      "suggest reaching out", "advise contacting", "recommend visiting",
      "not specified here", "not contained in", "not included in",
      "not part of", "outside the scope", "beyond the current",
      "not documented", "not recorded", "not presently available",
      "unable to locate", "cannot find", "don't have data",
      "not within our", "not among our", "not in our system",
      "would need to", "might want to", "should probably",
      "may need to", "would have to", "it's best to",
      "you may wish to", "consider reaching", "might consider",
      
      // Spanish phrases
      "no se encuentra", "no menciona", "no está disponible", "Traducción simulada:",
      "no aparece", "no se especifica", "no se aborda", 
      "no se cubre", "no se discute", "no hay información",
      "no tengo detalles", "consulte el sitio", "póngase en contacto",
      "busque en", "verifique con", "recomiendo", "debería consultar",
      "no estoy seguro", "no lo sé", "no puedo encontrar",
      "no tenemos datos", "no consta en", "no figura en",
      "diríjase a", "acuda a", "no se ha incluido",
      "no en los documentos", "no en nuestros registros",
      "no disponible actualmente", "sugiero contactar",
      "aconsejo comunicarse", "recomiendo visitar",
      "no especificado aquí", "no contenido en",
      "no incluido en", "no forma parte de",
      "fuera del alcance", "más allá de lo actual",
      "no documentado", "no registrado", "no disponible actualmente",
      "no puedo localizar", "no encuentro", "no tenemos información",
      "no dentro de nuestros", "no entre nuestros", "no en nuestro sistema",
      "tendría que", "quizás quiera", "probablemente debería",
      "podría necesitar", "tendría que", "sería mejor",
      "tal vez desee", "considere contactar", "podría considerar",
      "te recomendaría consultar", "comunicarte directamente",
      
      // Catch-all patterns
      /(sorry|lo siento).*(don't have|no tengo)/i,
      /(recommend|recomendar).*(contact|contactar)/i,
      /(refer|consulte).*(website|sitio)/i,
      /(not|no).*(in the|en los).*(documents|documentos)/i,
      /Traducción simulada:/i
    ];
    
    // Check for fallback phrases if not already triggered
    if (!needsWebSearch) {
      needsWebSearch = fallbackPhrases.some(phrase => {
        const answerText = typeof answerToCheck === 'string' ? answerToCheck.toLowerCase() : '';
        
        if (phrase instanceof RegExp) {
          return phrase.test(answerText);
        } else if (typeof phrase === 'string') {
          return answerText.includes(phrase.toLowerCase());
        }
        
        return false;
      });
    }

    let finalAnswer = answerToCheck;
    let source = 'rag-system';
    
    // Fall back to web search if needed
    if (needsWebSearch) {
      console.log('Fallback detected - activating web search agent');
      finalAnswer = await callWebSearchAgent(question, threadId, language);
      source = 'web-search-agent';
    }

    // Save to analytics collection
    await AnalyticsConversation.create({
      user: userId, // Save user reference
      question,
      answer: finalAnswer,
      language,
      source,
      threadId: threadId || 'auto-generated',
      timestamp: new Date()
    });

    return res.json({
      question,
      answer: finalAnswer,
      language,
      source,
      threadId: threadId || 'auto-generated',
      timestamp: new Date().toISOString()
    });

  } catch (err) {
    console.error('Error in /api/ask:', err);
    return res.status(500).json({ 
      error: language === 'es' 
        ? "Error procesando la pregunta. Por favor intente nuevamente." 
        : "Error processing question. Please try again.",
      details: err.message 
    });
  }
});

// Root endpoint
app.get('/', (req, res) => {
  res.send(`
    <h2>LAUSD High Schools Scraper API</h2>
    <h3>Available Endpoints:</h3>
    <ul>
      <li><strong>POST /api/ask</strong> - RAG with web search fallback</li>
      <li><strong>POST /api/web-search</strong> - Direct web search agent</li>
      <li><strong>GET/POST /api/conversations</strong> - Conversation management</li>
    </ul>
    <h3>Web Search Agent Usage:</h3>
    <code>
    POST /api/web-search<br>
    {<br>
    &nbsp;&nbsp;"question": "What are the latest LAUSD enrollment policies?",<br>
    &nbsp;&nbsp;"language": "en",<br>
    &nbsp;&nbsp;"threadId": "optional-thread-id"<br>
    }
    </code>
  `);
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`HTTP server running on http://192.168.1.78:${PORT}`);
});

// New RAG answer function with enhanced web search
async function getRagAnswer(question, language = 'en', userId = null, source = 'unknown') {
  console.log(`[${source}] getRagAnswer called with:`, question, language, userId);

  // Optionally log to a file or database for future AI improvement
  fs.appendFileSync('question_log.txt', JSON.stringify({
    timestamp: new Date().toISOString(),
    source,
    userId,
    question,
    language
  }) + '\n');

  const chunkCount = await DocumentChunk.countDocuments();
  console.log('Chunk count:', chunkCount);

  if (chunkCount === 0) {
    await initializeDatabase();
    console.log('Database initialized');
  }

  // Limit the number of chunks loaded to reduce memory usage
  const chunks = await DocumentChunk.find({}); // Adjust limit as needed
  console.log('Chunks found:', chunks.length);

  if (chunks.length === 0) {
    let answer = "No PDF documents found. Please ensure PDFs are in the correct directory.";
    let translated = answer;
    if (language && language !== 'en') {
      translated = await translateText(answer, language);
    }
    return { answer, translated };
  }

  const chunkEmbeddings = chunks.map(c => c.embedding);
  const questionEmbedding = await getEmbedding(question);
  console.log('Got question embedding');

  const similarities = chunkEmbeddings.map(e => cosineSimilarity(e, questionEmbedding));
  const topIndices = similarities
    .map((sim, idx) => ({ sim, idx }))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, 3)
    .map(obj => obj.idx);

  const ragContext = topIndices.map(idx =>
    `From ${chunks[idx].filename}:\n${chunks[idx].content}`
  ).join('\n\n---\n\n');

  const context = ragContext ? `Relevant LAUSD documents:\n\n${ragContext}` : '';
  console.log('Context built');

  if (!context) {
    let answer = "Sorry, I couldn't find relevant information in the LAUSD documents.";
    let translated = answer;
    if (language && language !== 'en') {
      translated = await translateText(answer, language);
    }
    return { answer, translated };
  }

  // Ask OpenAI with context
  console.log('Calling OpenAI...');
  const response = await makeOpenAIRequest({
    model: 'gpt-4o-mini',
    messages: [
      { role: 'system', content: 'You are a helpful assistant who answers questions using official LAUSD documents.' },
      { role: 'user', content: `${context}\n\nQuestion: ${question}` }
    ],
    temperature: 0.3
  });
  console.log('OpenAI response received');

  let answer = response.data.choices?.[0]?.message?.content || 'No answer generated.';

  // Check for misleading or unhelpful answers and trigger web search if needed
  const fallbackPhrases = [
    // English fallback phrases
    "do not specifically list resources",
    "couldn't find relevant information",
    "no relevant information",
    "not found in the provided documents",
    "not mentioned in the provided documents",
    "I recommend checking",
    "please visit",
    "you can find more information",
    "I suggest searching",
    "I'm not sure",
    "I don't know",
    // Spanish fallback phrases
    "no se enumeran específicamente los recursos",
    "no pude encontrar información relevante",
    "no hay información relevante",
    "no se encontró en los documentos proporcionados",
    "no se menciona en los documentos proporcionados",
    "recomiendo consultar",
    "por favor visite",
    "puede encontrar más información",
    "sugiero buscar",
    "no estoy seguro",
    "no lo sé"
  ];
  
  const isFallback = fallbackPhrases.some(phrase => {
  const answerText = typeof answer === 'string' ? answer.toLowerCase() : '';
  
  if (phrase instanceof RegExp) {
    return phrase.test(answerText);
  } else if (typeof phrase === 'string') {
    return answerText.includes(phrase.toLowerCase());
  }
  
  return false;
  });

  // If fallback detected, use the new web search agent
  if (isFallback) {
    console.log('Fallback detected, using Tavily web search agent...');
    try {
      answer = await callWebSearchAgent(question, null, language);
    } catch (webSearchError) {
      console.error('Web search agent failed:', webSearchError);
      answer = language === 'es' 
        ? "No pude encontrar información relevante en los documentos ni en la búsqueda web."
        : "I couldn't find relevant information in the documents or web search.";
    }
  }

  let translated = answer;
  if (language && language !== 'en') {
    translated = await translateText(answer, language);
  }
  return { answer, translated };
}

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('Shutting down gracefully...');
  if (mongoClient) {
    await mongoClient.close();
  }
  process.exit(0);
});

process.on('SIGINT', async () => {
  console.log('Shutting down gracefully...');
  if (mongoClient) {
    await mongoClient.close();
  }
  process.exit(0);
});

// New user signup endpoint
app.post('/api/signup', async (req, res) => {
  console.log('/api/signup route has been called!')
  const { username, email, password } = req.body;
  try {
    if (!username || !email || !password) {
      return res.status(400).json({ error: 'Todos los campos son obligatorios.' });
    }
    const existingUser = await User.findOne({ $or: [{ email }, { username }] });
    if (existingUser) {
      return res.status(409).json({ error: 'El usuario o correo ya existe.' });
    }
    const user = new User({ username, email, password });
    await user.save();
    res.status(201).json({
      message: 'Usuario creado exitosamente.',
      user: { id: user._id, username: user.username, email: user.email }
    });
  } catch (err) {
    res.status(500).json({ error: 'Error del servidor: ' + err.message });
  }
});

// New user login endpoint
app.post('/api/login', async (req, res) => {
  console.log('/api/login route has been called!')
  const { email, password } = req.body;
  try {
    if (!email || !password) {
      return res.status(400).json({ error: 'Correo y contraseña son obligatorios.' });
    }
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({ error: 'Credenciales inválidas.' });
    }
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(401).json({ error: 'Credenciales inválidas.' });
    }
    res.json({
      message: '¡Inicio de sesión exitoso!',
      user: { id: user._id, username: user.username, email: user.email }
    });
  } catch (err) {
    res.status(500).json({ error: 'Error del servidor: ' + err.message });
  }
});

// New Google login endpoint
const googleClient = new OAuth2Client(process.env.GOOGLE_CLIENT_ID);

app.post('/api/google-login', async (req, res) => {
  const { idToken } = req.body;
  try {
    // Verify token with Google
    const ticket = await googleClient.verifyIdToken({
      idToken,
      audience: process.env.GOOGLE_CLIENT_ID,
    });
    const payload = ticket.getPayload();
    const { email, name } = payload;

    // Check if user exists
    let user = await User.findOne({ email });
    if (!user) {
      // Create new user
      user = new User({
        username: name,
        email,
        password: Math.random().toString(36), // random, not used
      });
      await user.save();
    }

    // Respond with user info (including _id)
    res.json({
      message: 'Google login successful!',
      user: { id: user._id, username: user.username, email: user.email }
    });
  } catch (err) {
    res.status(401).json({ error: 'Invalid Google token', details: err.message });
  }
});