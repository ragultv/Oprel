import { API_BASE } from './api';

const MCP_BASE = () => `${API_BASE}/mcp`;

export interface Connector {
  id: string;
  name: string;
  description: string;
  category: string;
  icon: string;
  color: string;
  auth_type: string;
  auth_label: string;
  auth_help: string;
  auth_url?: string;
  tools_preview: string[];
  requires_package?: string;
  config_fields?: Array<{
    name: string;
    label: string;
    type: string;
    required?: boolean;
    help?: string;
  }>;
}

export interface ConnectorInstance {
  id: string;
  builtin_id: string;
  name: string;
  transport: string;
  config: Record<string, string>;
  enabled: boolean;
  status: 'disconnected' | 'connecting' | 'connected' | 'error';
  error?: string;
}

export const mcpApi = {
  // Catalog
  getCatalog: async (): Promise<{ connectors: Connector[] }> => {
    const res = await fetch(`${MCP_BASE()}/catalog`);
    if (!res.ok) throw new Error('Failed to fetch connector catalog');
    return res.json();
  },

  // Connector CRUD
  listConnectors: async (): Promise<{ connectors: ConnectorInstance[] }> => {
    const res = await fetch(`${MCP_BASE()}/connectors`);
    if (!res.ok) throw new Error('Failed to list connectors');
    return res.json();
  },

  addConnector: async (builtin_id: string, config: Record<string, string>, name?: string, enabled = true): Promise<{ connector: ConnectorInstance }> => {
    const res = await fetch(`${MCP_BASE()}/connectors`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ builtin_id, config, name, enabled }),
    });
    if (!res.ok) throw new Error('Failed to add connector');
    return res.json();
  },

  updateConnector: async (id: string, patch: Partial<ConnectorInstance>): Promise<{ connector: ConnectorInstance }> => {
    const res = await fetch(`${MCP_BASE()}/connectors/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    if (!res.ok) throw new Error('Failed to update connector');
    return res.json();
  },

  deleteConnector: async (id: string): Promise<void> => {
    const res = await fetch(`${MCP_BASE()}/connectors/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to delete connector');
  },

  // Lifecycle
  connectConnector: async (id: string): Promise<any> => {
    const res = await fetch(`${MCP_BASE()}/connectors/${encodeURIComponent(id)}/connect`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Failed to connect connector');
    return res.json();
  },

  disconnectConnector: async (id: string): Promise<any> => {
    const res = await fetch(`${MCP_BASE()}/connectors/${encodeURIComponent(id)}/disconnect`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Failed to disconnect connector');
    return res.json();
  },

  reconnectConnector: async (id: string): Promise<any> => {
    const res = await fetch(`${MCP_BASE()}/connectors/${encodeURIComponent(id)}/reconnect`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Failed to reconnect connector');
    return res.json();
  },

  testConnector: async (id: string): Promise<any> => {
    const res = await fetch(`${MCP_BASE()}/connectors/${encodeURIComponent(id)}/test`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error('Failed to test connector');
    return res.json();
  },

  // Tools & status
  getConnectorTools: async (id: string): Promise<{ tools: any[]; count: number }> => {
    const res = await fetch(`${MCP_BASE()}/connectors/${encodeURIComponent(id)}/tools`);
    if (!res.ok) throw new Error('Failed to fetch connector tools');
    return res.json();
  },

  getStatus: async (): Promise<{ total_connectors: number; connected: number; disconnected: number; total_tools: number; live_connector_ids: string[] }> => {
    const res = await fetch(`${MCP_BASE()}/status`);
    if (!res.ok) throw new Error('Failed to fetch MCP status');
    return res.json();
  },
};
