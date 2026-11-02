// Paste inside the ChatView component, next to `const [toolsMenuOpen, setToolsMenuOpen]`.
// Needs: import { Switch } from '@/components/ui/switch'; import { MCP, type Connector } from '@/services/mcp'

  const [moreToolsOpen, setMoreToolsOpen] = useState(false)
  const [chatConnectors, setChatConnectors] = useState<Connector[]>([])

  useEffect(() => {
    if (!toolsMenuOpen) return
    MCP.connectors().then(setChatConnectors).catch(() => setChatConnectors([]))
  }, [toolsMenuOpen])

  useEffect(() => {
    if (!toolsMenuOpen) setMoreToolsOpen(false)
  }, [toolsMenuOpen])

  const connectorIssues = chatConnectors.filter((c) => c.enabled && c.status !== "connected").length

  const toggleChatConnector = async (c: Connector, on: boolean) => {
    setChatConnectors((list) => list.map((x) => (x.id === c.id ? { ...x, use_in_chat: on } : x)))
    try {
      await MCP.patch(c.id, { use_in_chat: on })
    } catch {
      setChatConnectors((list) => list.map((x) => (x.id === c.id ? { ...x, use_in_chat: !on } : x)))
      toast({ title: "Could not change connector", description: "Please try again.", variant: "destructive" })
    }
  }
