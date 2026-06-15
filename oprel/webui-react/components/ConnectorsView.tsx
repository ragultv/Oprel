"use client"

import { useState, useEffect, useMemo } from "react"
import { Plug, CheckCircle2, Loader2, AlertCircle, RefreshCw, Trash2, Key, Info, ExternalLink, Search } from "lucide-react"
import { mcpApi, Connector, ConnectorInstance } from "@/services/mcp"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter, DialogClose } from "@/components/ui/dialog"

function ConnectorIcon({ icon, size = 24, className = "" }: { icon: string, size?: number, className?: string }) {
  const map: Record<string, string> = {
    'figma': 'figma',
    'canva': 'canva',
    'notion': 'notion',
    'google-drive': 'googledrive',
    'word': 'microsoftword',
    'excel': 'microsoftexcel',
    'powerpoint': 'microsoftpowerpoint',
    'gmail': 'gmail',
    'google-calendar': 'googlecalendar',
    'github': 'github',
  }
  
  if (icon === 'search') {
    return <Search size={size} className={`text-white ${className}`} />
  }
  
  const mapped = map[icon] || icon
  return <img src={`https://cdn.simpleicons.org/${mapped}/white`} alt={icon} className={`object-contain ${className}`} style={{ width: size, height: size }} onError={(e) => e.currentTarget.style.display = 'none'} />
}

export function ConnectorsView() {
  const [catalog, setCatalog] = useState<Connector[]>([])
  const [instances, setInstances] = useState<ConnectorInstance[]>([])
  
  const [dialogOpen, setDialogOpen] = useState(false)
  const [activeConnector, setActiveConnector] = useState<Connector | null>(null)
  const [activeInstance, setActiveInstance] = useState<ConnectorInstance | null>(null)
  
  const [configValues, setConfigValues] = useState<Record<string, string>>({})
  const [showToken, setShowToken] = useState(false)
  
  const [loading, setLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  
  const [toolsPreview, setToolsPreview] = useState<any[]>([])
  const [toolsCount, setToolsCount] = useState(0)

  // Fetch initial data
  const fetchData = async () => {
    try {
      const [catRes, instRes] = await Promise.all([
        mcpApi.getCatalog(),
        mcpApi.listConnectors()
      ])
      setCatalog(catRes.connectors)
      setInstances(instRes.connectors)
      setLoading(false)
    } catch (err) {
      console.error("Failed to fetch connectors data", err)
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  // Poll instances
  useEffect(() => {
    if (loading) return;
    const interval = setInterval(async () => {
      try {
        const instRes = await mcpApi.listConnectors()
        setInstances(instRes.connectors)
        
        // Update active instance if dialog is open
        if (dialogOpen && activeInstance) {
          const updated = instRes.connectors.find((i: any) => i.id === activeInstance.id)
          if (updated) setActiveInstance(updated)
        }
      } catch (err) {
        // Silently fail polling
      }
    }, 3000)
    
    return () => clearInterval(interval)
  }, [loading, dialogOpen, activeInstance])

  // Load tools when connected
  useEffect(() => {
    if (dialogOpen && activeInstance?.status === 'connected') {
      mcpApi.getConnectorTools(activeInstance.id).then(res => {
        setToolsPreview(res.tools.slice(0, 4))
        setToolsCount(res.count)
      }).catch(() => {})
    }
  }, [dialogOpen, activeInstance?.status])

  // Load tools when connected

  const openConnectDialog = (connector: Connector) => {
    setActiveConnector(connector)
    setActiveInstance(null)
    setConfigValues({})
    setShowToken(false)
    setActionError(null)
    setToolsPreview([])
    setToolsCount(0)
    setDialogOpen(true)
  }

  const openManageDialog = (connector: Connector, instance: ConnectorInstance) => {
    setActiveConnector(connector)
    setActiveInstance(instance)
    setConfigValues({})
    setShowToken(false)
    setActionError(null)
    setToolsPreview([])
    setToolsCount(0)
    setDialogOpen(true)
  }

  const handleConnect = async () => {
    if (!activeConnector) return
    setActionLoading(true)
    setActionError(null)
    
    try {
      const res = await mcpApi.addConnector(activeConnector.id, configValues, activeConnector.name, true)
      setActiveInstance(res.connector)
      await fetchData() // Refresh lists
    } catch (err: any) {
      setActionError(err.message || "Failed to connect")
    } finally {
      setActionLoading(false)
    }
  }

  const handleConnectOAuth = async () => {
    if (!activeConnector) return
    setActionLoading(true)
    setActionError(null)
    
    try {
      let instanceId = activeInstance?.id;
      
      // If no instance exists yet, create one with empty config
      if (!instanceId) {
          const createRes = await mcpApi.addConnector(activeConnector.id, {}, activeConnector.name, true);
          instanceId = createRes.connector.id;
      }
      
      const res = await fetch(`/api/mcp/connectors/${instanceId}/oauth/start`);
      if (!res.ok) {
          const errData = await res.json().catch(() => null);
          throw new Error(errData?.detail || "Failed to start OAuth flow");
      }
      const { auth_url } = await res.json();
      
      const popup = window.open(auth_url, 'oauth', 'width=600,height=700');
      
      const listener = async (e: MessageEvent) => {
          if (e.data?.type === 'oauth_success') {
              popup?.close();
              window.removeEventListener('message', listener);
              await fetchData();
              setActionLoading(false);
          }
      };
      window.addEventListener('message', listener);
    } catch (err: any) {
      setActionError(err.message || "Failed to connect via OAuth")
      setActionLoading(false)
    }
  }

  const handleDisconnect = async () => {
    if (!activeInstance) return
    setActionLoading(true)
    try {
      await mcpApi.disconnectConnector(activeInstance.id)
      await fetchData()
    } catch (err: any) {
      setActionError(err.message || "Failed to disconnect")
    } finally {
      setActionLoading(false)
    }
  }

  const handleReconnect = async () => {
    if (!activeInstance) return
    setActionLoading(true)
    try {
      await mcpApi.reconnectConnector(activeInstance.id)
      await fetchData()
    } catch (err: any) {
      setActionError(err.message || "Failed to reconnect")
    } finally {
      setActionLoading(false)
    }
  }

  const handleRemove = async () => {
    if (!activeInstance) return
    setActionLoading(true)
    try {
      await mcpApi.deleteConnector(activeInstance.id)
      setDialogOpen(false)
      await fetchData()
    } catch (err: any) {
      setActionError(err.message || "Failed to remove")
    } finally {
      setActionLoading(false)
    }
  }

  const handleTest = async () => {
    if (!activeInstance) return
    setActionLoading(true)
    try {
      await mcpApi.testConnector(activeInstance.id)
      alert("Test successful!")
    } catch (err: any) {
      setActionError(err.message || "Test failed")
    } finally {
      setActionLoading(false)
    }
  }

  return (
    <div className="h-full flex flex-col bg-[#111] overflow-hidden">
      {/* Header */}
      <div className="px-10 py-10 shrink-0 bg-gradient-to-b from-[#1a1a1a] to-[#111] border-b border-border flex items-end justify-between">
        <div>
          <h1 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-primary to-primary/60 tracking-tight flex items-center gap-3">
            <Plug size={28} className="text-primary" /> Integrations
          </h1>
          <p className="text-base text-muted-foreground mt-2 ml-10">
            Connect external tools and supercharge your local models.
          </p>
        </div>
        <div className="px-3 py-1.5 rounded-full bg-secondary text-xs font-semibold text-muted-foreground flex items-center gap-2 border border-border">
          <div className="w-2 h-2 rounded-full bg-green-500"></div>
          {instances.filter(i => i.status === 'connected').length} / {catalog.length} Connected
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Content */}
        <div className="flex-1 overflow-y-auto p-8">
          {loading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="animate-spin text-muted-foreground" size={32} />
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {catalog.map(connector => {
                const instance = instances.find(i => i.builtin_id === connector.id)
                const isConnected = instance?.status === 'connected'
                const isConnecting = instance?.status === 'connecting'
                const isError = instance?.status === 'error'

                return (
                  <div key={connector.id} onClick={() => instance ? openManageDialog(connector, instance) : openConnectDialog(connector)} className="bg-[#171717] border border-border/50 rounded-2xl p-6 flex flex-col items-center text-center transition-all hover:border-primary/50 hover:bg-[#1a1a1a] hover:scale-[1.02] hover:shadow-2xl hover:shadow-primary/10 relative overflow-hidden group cursor-pointer">
                    <div 
                        className="w-20 h-20 rounded-2xl flex items-center justify-center shadow-xl mb-4 transition-transform group-hover:scale-110"
                        style={{ backgroundColor: connector.color }}
                      >
                        <ConnectorIcon icon={connector.icon} size={40} />
                      </div>
                      
                      {instance && (
                        <div className={`absolute top-4 right-4 px-2 py-0.5 rounded-full text-[9px] font-bold uppercase tracking-wider flex items-center gap-1
                          ${isConnected ? 'bg-green-500/10 text-green-500 border border-green-500/20' : ''}
                          ${isConnecting ? 'bg-amber-500/10 text-amber-500 border border-amber-500/20' : ''}
                          ${isError ? 'bg-red-500/10 text-red-500 border border-red-500/20' : ''}
                          ${instance.status === 'disconnected' ? 'bg-secondary text-muted-foreground' : ''}
                        `}>
                          {isConnected && <CheckCircle2 size={10} />}
                          {isConnecting && <Loader2 size={10} className="animate-spin" />}
                          {isError && <AlertCircle size={10} />}
                          {instance.status}
                        </div>
                      )}
                    
                    <h3 className="font-bold text-lg text-foreground tracking-tight">{connector.name}</h3>
                    <p className="text-xs text-muted-foreground mt-2 line-clamp-2 min-h-[32px] opacity-80 group-hover:opacity-100 transition-opacity">{connector.description}</p>
                    
                    {connector.requires_package && (
                      <div className="mt-4 text-[10px] text-amber-500/80 bg-amber-500/10 px-2 py-1 rounded border border-amber-500/20 w-fit">
                        Requires: {connector.requires_package}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* Config Dialog */}
      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="sm:max-w-md bg-[#1a1a1a] border-border">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-3 text-xl">
              {activeConnector && (
                <div 
                  className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0 shadow-sm"
                  style={{ backgroundColor: activeConnector.color }}
                >
                  <ConnectorIcon icon={activeConnector.icon} />
                </div>
              )}
              {activeConnector?.name}
              
              {activeInstance?.status === 'connected' && (
                <span className="ml-auto text-xs px-2 py-1 bg-green-500/10 text-green-500 border border-green-500/20 rounded-full flex items-center gap-1">
                  <CheckCircle2 size={12} /> Connected
                </span>
              )}
            </DialogTitle>
            <DialogDescription>
              {activeConnector?.description}
            </DialogDescription>
          </DialogHeader>
          
          <div className="py-4">
            {actionError && (
              <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-500 text-xs flex items-start gap-2">
                <AlertCircle size={14} className="shrink-0 mt-0.5" />
                <span>{actionError}</span>
              </div>
            )}
            
            {/* Show config form if not connected (or no instance) */}
            {(!activeInstance || (activeInstance.status !== 'connected' && activeInstance.status !== 'connecting')) && activeConnector && (
              <div className="space-y-4">
                {activeConnector.auth_type === 'oauth_pkce' ? (
                  <div className="space-y-4 text-center py-4">
                    <div className="mx-auto w-12 h-12 rounded-xl flex items-center justify-center mb-4 shadow-lg" style={{ backgroundColor: activeConnector.color }}>
                      <ConnectorIcon icon={activeConnector.icon} />
                    </div>
                    <p className="text-sm text-foreground">
                      {activeConnector.auth_help || `Authenticate with ${activeConnector.name} to continue.`}
                    </p>
                    {activeConnector.auth_url && (
                        <a href={activeConnector.auth_url} target="_blank" rel="noreferrer" className="text-xs text-primary hover:underline block mt-2">
                            View API Documentation
                        </a>
                    )}
                  </div>
                ) : activeConnector.auth_type === 'api_key' ? (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-semibold text-foreground flex items-center gap-1.5">
                        <Key size={12} className="text-muted-foreground" /> {activeConnector.auth_label || "API Key"}
                      </label>
                      {activeConnector.auth_help && (
                        <a href={activeConnector.auth_url || "#"} target="_blank" rel="noreferrer" className="text-[10px] text-primary hover:underline flex items-center gap-1">
                          Get Key <ExternalLink size={10} />
                        </a>
                      )}
                    </div>
                    <div className="relative">
                      <input 
                        type={showToken ? "text" : "password"} 
                        placeholder="Paste your token here"
                        className="w-full bg-[#111] border border-border rounded-lg px-3 py-2.5 text-sm text-foreground focus:border-primary/50 transition-colors pr-12"
                        value={configValues.api_key || ''}
                        onChange={(e) => setConfigValues({...configValues, api_key: e.target.value})}
                      />
                      <button 
                        type="button"
                        onClick={() => setShowToken(!showToken)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] font-bold text-muted-foreground hover:text-foreground"
                      >
                        {showToken ? "HIDE" : "SHOW"}
                      </button>
                    </div>
                    {activeConnector.auth_help && (
                      <p className="text-[10px] text-muted-foreground mt-1.5 flex items-start gap-1">
                        <Info size={12} className="shrink-0 mt-0.5" /> {activeConnector.auth_help}
                      </p>
                    )}
                  </div>
                ) : activeConnector.config_fields ? (
                  activeConnector.config_fields.map((field, idx) => (
                    <div key={idx} className="space-y-1.5">
                      <label className="text-xs font-semibold text-foreground">
                        {field.label} {field.required && <span className="text-red-500">*</span>}
                      </label>
                      <input 
                        type={field.type === 'password' && !showToken ? "password" : "text"} 
                        placeholder={field.label}
                        className="w-full bg-[#111] border border-border rounded-lg px-3 py-2 text-sm text-foreground focus:border-primary/50 transition-colors"
                        value={configValues[field.name] || ''}
                        onChange={(e) => setConfigValues({...configValues, [field.name]: e.target.value})}
                      />
                      {field.help && <p className="text-[10px] text-muted-foreground">{field.help}</p>}
                    </div>
                  ))
                ) : null}
                
                {activeConnector.requires_package && (
                  <div className="p-3 bg-amber-500/10 border border-amber-500/20 rounded-lg text-amber-500 text-xs flex items-start gap-2">
                    <AlertCircle size={14} className="shrink-0 mt-0.5" />
                    <span>Requires <strong>{activeConnector.requires_package}</strong> to be installed on your system.</span>
                  </div>
                )}
              </div>
            )}
            
            {/* Show loading state */}
            {activeInstance?.status === 'connecting' && (
              <div className="py-8 flex flex-col items-center justify-center text-center">
                <RefreshCw size={32} className="animate-spin text-primary mb-4" />
                <h4 className="text-sm font-bold text-foreground">Connecting to {activeConnector?.name}...</h4>
                <p className="text-xs text-muted-foreground mt-2">Authenticating and fetching available tools</p>
              </div>
            )}
            
            {/* Show connected state tools */}
            {activeInstance?.status === 'connected' && (
              <div className="space-y-4">
                <div className="p-4 bg-[#111] rounded-lg border border-border">
                  <h4 className="text-xs font-bold text-foreground mb-3 flex items-center justify-between">
                    <span>Available Tools</span>
                    <span className="px-2 py-0.5 bg-secondary rounded text-[10px] text-muted-foreground">{toolsCount} total</span>
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {toolsPreview.map((tool: any, idx: number) => (
                      <div key={idx} className="px-2 py-1 rounded bg-secondary text-xs text-foreground font-mono">
                        {tool.name}
                      </div>
                    ))}
                    {toolsCount > 4 && (
                      <div className="px-2 py-1 rounded text-xs text-muted-foreground">
                        +{toolsCount - 4} more
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
          
          <DialogFooter className="sm:justify-between border-t border-border pt-4 mt-2">
            {!activeInstance ? (
              <>
                <DialogClose asChild>
                  <button className="px-4 py-2 bg-secondary text-foreground text-sm font-semibold rounded-lg hover:bg-secondary/80">
                    Cancel
                  </button>
                </DialogClose>
                <button 
                  onClick={activeConnector?.auth_type === 'oauth_pkce' ? handleConnectOAuth : handleConnect}
                  disabled={actionLoading}
                  className="px-4 py-2 bg-primary text-primary-foreground text-sm font-semibold rounded-lg hover:bg-primary/90 flex items-center gap-2"
                >
                  {actionLoading ? <Loader2 size={16} className="animate-spin" /> : null}
                  {activeConnector?.auth_type === 'oauth_pkce' ? `Authenticate` : 'Connect'}
                </button>
              </>
            ) : activeInstance.status === 'connected' ? (
              <>
                <button 
                  onClick={handleDisconnect}
                  disabled={actionLoading}
                  className="px-4 py-2 bg-secondary text-foreground text-sm font-semibold rounded-lg hover:bg-secondary/80 flex items-center gap-2"
                >
                  Disconnect
                </button>
                <div className="flex gap-2">
                  <button 
                    onClick={handleTest}
                    disabled={actionLoading}
                    className="px-4 py-2 bg-secondary text-foreground text-sm font-semibold rounded-lg hover:bg-secondary/80 flex items-center gap-2"
                  >
                    Test
                  </button>
                  <button 
                    onClick={handleReconnect}
                    disabled={actionLoading}
                    className="px-4 py-2 bg-primary text-primary-foreground text-sm font-semibold rounded-lg hover:bg-primary/90 flex items-center gap-2"
                  >
                    {actionLoading ? <Loader2 size={16} className="animate-spin" /> : null}
                    Reconnect
                  </button>
                </div>
              </>
            ) : (
              <>
                <button 
                  onClick={handleRemove}
                  disabled={actionLoading}
                  className="px-4 py-2 bg-destructive text-white text-sm font-semibold rounded-lg hover:bg-destructive/90 flex items-center gap-2"
                >
                  Remove
                </button>
                <button 
                  onClick={handleReconnect}
                  disabled={actionLoading}
                  className="px-4 py-2 bg-primary text-primary-foreground text-sm font-semibold rounded-lg hover:bg-primary/90 flex items-center gap-2"
                >
                  {actionLoading ? <Loader2 size={16} className="animate-spin" /> : null}
                  Reconnect
                </button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
