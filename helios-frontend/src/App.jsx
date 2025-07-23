import React, { useState, useEffect, useRef } from 'react'
import {
  AppShell,
  Header,
  Title,
  Container,
  Stack,
  Paper,
  Textarea,
  Button,
  Group,
  Text,
  Badge,
  Loader,
  ScrollArea,
  ActionIcon,
  Tooltip,
  Alert,
  Divider
} from '@mantine/core'
import {
  IconSend,
  IconClearAll,
  IconActivity,
  IconAlertCircle,
  IconCopy,
  IconCheck
} from '@tabler/icons-react'
import { notifications } from '@mantine/notifications'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import axios from 'axios'

// Configure axios
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 300000, // 5 minutes
})

// Custom markdown renderer with syntax highlighting
const MarkdownRenderer = ({ children }) => {
  return (
    <ReactMarkdown
      components={{
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || '')
          return !inline && match ? (
            <SyntaxHighlighter
              style={oneDark}
              language={match[1]}
              PreTag="div"
              {...props}
            >
              {String(children).replace(/\n$/, '')}
            </SyntaxHighlighter>
          ) : (
            <code className={className} {...props}>
              {children}
            </code>
          )
        }
      }}
    >
      {children}
    </ReactMarkdown>
  )
}

// Message component
const Message = ({ message, onCopy }) => {
  const [copied, setCopied] = useState(false)

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
    onCopy()
  }

  return (
    <Paper
      p="md"
      withBorder
      style={{
        backgroundColor: message.sender === 'user' ? '#1a1b1e' : '#0c0d0e',
        borderLeft: `4px solid ${message.sender === 'user' ? '#339af0' : '#51cf66'}`
      }}
    >
      <Group justify="space-between" mb="xs">
        <Group>
          <Badge
            color={message.sender === 'user' ? 'blue' : 'green'}
            variant="filled"
          >
            {message.sender === 'user' ? 'You' : 'Helios'}
          </Badge>
          {message.model_used && (
            <Badge variant="outline" size="xs">
              {message.model_used}
            </Badge>
          )}
        </Group>
        <Tooltip label={copied ? 'Copied!' : 'Copy message'}>
          <ActionIcon
            variant="subtle"
            onClick={handleCopy}
            color={copied ? 'green' : 'gray'}
          >
            {copied ? <IconCheck size={16} /> : <IconCopy size={16} />}
          </ActionIcon>
        </Tooltip>
      </Group>
      
      {message.sender === 'user' ? (
        <Text style={{ whiteSpace: 'pre-wrap' }}>{message.content}</Text>
      ) : (
        <MarkdownRenderer>{message.content}</MarkdownRenderer>
      )}
      
      {message.timestamp && (
        <Text size="xs" c="dimmed" mt="xs">
          {new Date(message.timestamp).toLocaleString()}
        </Text>
      )}
    </Paper>
  )
}

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [status, setStatus] = useState('idle')
  const [healthStatus, setHealthStatus] = useState(null)
  
  const scrollAreaRef = useRef(null)
  const textareaRef = useRef(null)

  // Check service health on startup
  useEffect(() => {
    checkHealth()
  }, [])

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTo({ top: scrollAreaRef.current.scrollHeight, behavior: 'smooth' })
    }
  }, [messages])

  const checkHealth = async () => {
    try {
      const response = await api.get('/health')
      setHealthStatus(response.data)
    } catch (error) {
      console.error('Health check failed:', error)
      setHealthStatus({ status: 'error', services: {} })
    }
  }

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage = {
      sender: 'user',
      content: input,
      timestamp: new Date().toISOString(),
    }

    // Add user message immediately
    setMessages(prev => [...prev, userMessage])
    
    // Parse input to separate logs from question
    const lines = input.trim().split('\n')
    let logs = null
    let question = input

    // Simple heuristic: if input has multiple lines and looks like logs, separate them
    if (lines.length > 5) {
      // Assume it's logs with a question at the end
      const potentialQuestion = lines[lines.length - 1]
      if (potentialQuestion.includes('?') || potentialQuestion.toLowerCase().includes('what') || 
          potentialQuestion.toLowerCase().includes('why') || potentialQuestion.toLowerCase().includes('how')) {
        logs = lines.slice(0, -1).join('\n')
        question = potentialQuestion
      }
    }

    setInput('')
    setIsLoading(true)
    setStatus('analyzing')

    try {
      const response = await api.post('/chat', {
        logs,
        question,
        session_id: sessionId,
      })

      const aiMessage = {
        sender: 'assistant',
        content: response.data.response,
        model_used: response.data.model_used,
        timestamp: response.data.timestamp,
      }

      setMessages(prev => [...prev, aiMessage])
      
      // Store session ID for conversation continuity
      if (!sessionId) {
        setSessionId(response.data.session_id)
      }

      setStatus('idle')
    } catch (error) {
      console.error('Error sending message:', error)
      
      const errorMessage = {
        sender: 'system',
        content: `Error: ${error.response?.data?.detail || error.message || 'Failed to get response from Helios'}`,
        timestamp: new Date().toISOString(),
      }
      
      setMessages(prev => [...prev, errorMessage])
      setStatus('error')
      
      notifications.show({
        title: 'Error',
        message: 'Failed to get response from Helios. Please try again.',
        color: 'red',
        icon: <IconAlertCircle size={18} />,
      })
    } finally {
      setIsLoading(false)
    }
  }

  const clearConversation = () => {
    setMessages([])
    setSessionId(null)
    setStatus('idle')
    notifications.show({
      title: 'Conversation Cleared',
      message: 'Started a new conversation session.',
      color: 'blue',
    })
  }

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault()
      sendMessage()
    }
  }

  const handleCopyMessage = () => {
    notifications.show({
      title: 'Copied!',
      message: 'Message copied to clipboard.',
      color: 'green',
      autoClose: 2000,
    })
  }

  const getStatusColor = () => {
    switch (status) {
      case 'analyzing': return 'blue'
      case 'error': return 'red'
      default: return 'green'
    }
  }

  const getStatusText = () => {
    switch (status) {
      case 'analyzing': return 'Analyzing...'
      case 'error': return 'Error'
      default: return 'Ready'
    }
  }

  return (
    <AppShell>
      <AppShell.Header p="md">
        <Group justify="space-between">
          <Group>
            <IconActivity size={28} color="#51cf66" />
            <Title order={2} c="white">
              Helios
            </Title>
            <Badge variant="outline" size="sm">
              AI-Powered Root Cause Analysis
            </Badge>
          </Group>
          
          <Group>
            <Group gap="xs">
              <Text size="sm" c="dimmed">Status:</Text>
              <Badge color={getStatusColor()} variant="filled">
                {getStatusText()}
              </Badge>
            </Group>
            
            {healthStatus && (
              <Badge 
                color={healthStatus.status === 'healthy' ? 'green' : 'yellow'} 
                variant="outline"
              >
                Services: {healthStatus.status}
              </Badge>
            )}
            
            <Tooltip label="Clear conversation">
              <ActionIcon
                variant="outline"
                onClick={clearConversation}
                disabled={messages.length === 0}
              >
                <IconClearAll size={18} />
              </ActionIcon>
            </Tooltip>
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Main>
        <Container size="xl" h="100%">
          <Stack h="100%" spacing="md" py="md">
            {/* Welcome message */}
            {messages.length === 0 && (
              <Alert icon={<IconActivity size={16} />} title="Welcome to Helios!" color="blue">
                <Text>
                  I'm your AI-powered root cause analysis assistant. You can:
                </Text>
                <Text mt="xs" component="ul" style={{ paddingLeft: '1rem' }}>
                  <li>Paste log files and ask "What went wrong?"</li>
                  <li>Describe system issues and get structured analysis</li>
                  <li>Ask questions about observability best practices</li>
                </Text>
                <Text mt="xs" size="sm" c="dimmed">
                  Tip: Use Ctrl+Enter (Cmd+Enter on Mac) to send your message.
                </Text>
              </Alert>
            )}

            {/* Messages */}
            <ScrollArea 
              flex={1} 
              ref={scrollAreaRef}
              type="scroll"
              offsetScrollbars
            >
              <Stack spacing="md">
                {messages.map((message, index) => (
                  <Message 
                    key={index} 
                    message={message} 
                    onCopy={handleCopyMessage}
                  />
                ))}
                
                {isLoading && (
                  <Paper p="md" withBorder style={{ backgroundColor: '#0c0d0e' }}>
                    <Group>
                      <Loader size="sm" color="green" />
                      <Text c="dimmed">Helios is analyzing...</Text>
                    </Group>
                  </Paper>
                )}
              </Stack>
            </ScrollArea>

            {/* Input area */}
            <Paper p="md" withBorder>
              <Stack spacing="md">
                <Textarea
                  ref={textareaRef}
                  placeholder="Paste your logs here or ask a question about system issues..."
                  value={input}
                  onChange={(event) => setInput(event.currentTarget.value)}
                  onKeyDown={handleKeyPress}
                  minRows={3}
                  maxRows={10}
                  autosize
                  disabled={isLoading}
                />
                
                <Group justify="space-between">
                  <Text size="sm" c="dimmed">
                    Ctrl+Enter to send • {input.length} characters
                  </Text>
                  
                  <Button
                    onClick={sendMessage}
                    disabled={!input.trim() || isLoading}
                    leftSection={<IconSend size={16} />}
                    loading={isLoading}
                  >
                    {isLoading ? 'Analyzing...' : 'Send'}
                  </Button>
                </Group>
              </Stack>
            </Paper>
          </Stack>
        </Container>
      </AppShell.Main>
    </AppShell>
  )
}

export default App 