"use client"

import { useState, useEffect } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import { Users, Plus, Hash, Settings, Trash2, UserPlus, Cpu, Cloud, Crown, ChevronDown } from "lucide-react"
import { GroupsAPI, type Group, type GroupMember } from "@/services/api"
import { parseProviderModelId } from "@/services/providers"
import { useApp } from "@/services/context"
import { cn } from "@/services/utils"
import { useToast } from "@/components/ui/use-toast"

export function GroupSidebar() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const activeGroupId = searchParams.get("groupId")
  const { models } = useApp()
  const { toast } = useToast()

  const [groups, setGroups] = useState<Group[]>([])
  const [loading, setLoading] = useState(true)

  // Modals state
  const [createGroupOpen, setCreateGroupOpen] = useState(false)
  const [newGroupName, setNewGroupName] = useState("")

  const [addMemberOpen, setAddMemberOpen] = useState(false)
  const [newMemberName, setNewMemberName] = useState("")
  const [newMemberModelId, setNewMemberModelId] = useState("")
  const [modelDropdownOpen, setModelDropdownOpen] = useState(false)
  const [newMemberRole, setNewMemberRole] = useState("")
  const [newMemberModerator, setNewMemberModerator] = useState(false)

  const fetchGroups = async () => {
    try {
      const data = await GroupsAPI.fetchGroups()
      setGroups(data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchGroups()
  }, [])

  const activeGroup = groups.find((g) => g.id === activeGroupId)

  const handleCreateGroup = async () => {
    if (!newGroupName.trim()) return
    try {
      const newGroup = await GroupsAPI.createGroup({ name: newGroupName })
      await fetchGroups()
      setCreateGroupOpen(false)
      setNewGroupName("")
      router.push(`/groups?groupId=${newGroup.id}`)
    } catch (err: any) {
      toast({ title: "Error", description: err.message, variant: "destructive" })
    }
  }

  const handleDeleteGroup = async (id: string) => {
    if (!confirm("Delete this group?")) return
    try {
      await GroupsAPI.deleteGroup(id)
      await fetchGroups()
      if (activeGroupId === id) {
        router.push("/groups")
      }
    } catch (err: any) {
      toast({ title: "Error", description: err.message, variant: "destructive" })
    }
  }

  const handleAddMember = async () => {
    if (!activeGroupId || !newMemberName || !newMemberModelId) return
    
    let kind = "local"
    
    const selectedModel = models.find(m => m.id === newMemberModelId)
    if (selectedModel?.category === 'external') {
      kind = "cloud"
    }
    
    try {
      await GroupsAPI.addMember(activeGroupId, {
        kind: kind,
        provider_id: undefined,
        model_id: newMemberModelId,
        display_name: newMemberName,
        role_description: newMemberRole,
        is_moderator: newMemberModerator ? 1 : 0,
        priority_order: activeGroup?.members?.length || 0,
      })
      await fetchGroups()
      setAddMemberOpen(false)
      setNewMemberName("")
      setNewMemberModelId("")
      setNewMemberRole("")
      setNewMemberModerator(false)
    } catch (err: any) {
      toast({ title: "Error", description: err.message, variant: "destructive" })
    }
  }

  const handleRemoveMember = async (groupId: string, memberId: string) => {
    if (!confirm("Remove this member?")) return
    try {
      await GroupsAPI.removeMember(groupId, memberId)
      await fetchGroups()
    } catch (err: any) {
      toast({ title: "Error", description: err.message, variant: "destructive" })
    }
  }

  // Filter models
  const localModels = models.filter((m) => (m.status === "loaded" || m.status === "available" || m.downloaded) && m.category !== 'external')
  const cloudModels = models.filter((m) => m.category === 'external')

  const selectedModelInfo = models.find(m => m.id === newMemberModelId)

  // Dummy preset colors for providers just like ChatView
  const PROVIDER_COLORS: Record<string, string> = {
    openai: "#10a37f",
    anthropic: "#d97757",
    google: "#4285f4",
    groq: "#f55036",
    ollama_compat: "#000000",
  };

  return (
    <div className="w-64 border-r border-border bg-[#171717] flex flex-col h-full shrink-0">
      <div className="p-4 border-b border-border flex items-center justify-between">
        <h2 className="font-bold text-sm tracking-tight flex items-center gap-2 text-foreground">
          <Users size={16} className="text-primary" />
          Groups
        </h2>
        <button
          onClick={() => setCreateGroupOpen(true)}
          className="p-1.5 hover:bg-secondary rounded-md transition-colors text-muted-foreground hover:text-foreground"
        >
          <Plus size={16} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {loading ? (
          <div className="text-xs text-muted-foreground p-2">Loading groups...</div>
        ) : groups.length === 0 ? (
          <div className="text-xs text-muted-foreground p-2 text-center mt-4">No groups created.</div>
        ) : (
          groups.map((group) => (
            <div key={group.id} className="mb-2">
              <div
                onClick={() => router.push(`/groups?groupId=${group.id}`)}
                className={cn(
                  "group/item flex items-center justify-between p-2 rounded-md cursor-pointer transition-colors text-sm",
                  activeGroupId === group.id
                    ? "bg-secondary text-foreground"
                    : "text-muted-foreground hover:bg-secondary/50 hover:text-foreground"
                )}
              >
                <div className="flex items-center gap-2 truncate">
                  <Hash size={14} className="opacity-70" />
                  <span className="truncate">{group.name}</span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleDeleteGroup(group.id)
                  }}
                  className="opacity-0 group-hover/item:opacity-100 p-1 hover:text-destructive transition-opacity"
                >
                  <Trash2 size={12} />
                </button>
              </div>

              {/* Show members if this group is active */}
              {activeGroupId === group.id && (
                <div className="ml-6 pl-2 border-l border-border/50 mt-1 space-y-1">
                  <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2 mt-2 flex justify-between items-center pr-2">
                    Members ({group.members?.length || 0})
                    <button
                      onClick={() => setAddMemberOpen(true)}
                      className="text-primary hover:text-primary/80 flex items-center gap-1"
                    >
                      <UserPlus size={10} /> Add
                    </button>
                  </div>
                  {group.members?.map((member) => (
                    <div key={member.id} className="flex items-center justify-between text-xs py-1 pr-2 group/member">
                      <div className="flex items-center gap-1.5 text-muted-foreground truncate">
                        {member.kind === "local" ? <Cpu size={10} /> : <Cloud size={10} />}
                        <span className={cn("truncate", member.is_moderator && "text-amber-500 font-medium")}>
                          {member.display_name}
                        </span>
                        {member.is_moderator === 1 && <Crown size={10} className="text-amber-500 ml-0.5" />}
                      </div>
                      <button
                        onClick={() => handleRemoveMember(group.id, member.id)}
                        className="opacity-0 group-hover/member:opacity-100 hover:text-destructive text-muted-foreground transition-opacity"
                      >
                        <Trash2 size={10} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Create Group Modal */}
      {createGroupOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-[#1e1e1e] border border-border rounded-xl p-6 w-96 shadow-2xl animate-in fade-in zoom-in-95">
            <h3 className="text-sm font-bold text-foreground mb-4">Create New Group</h3>
            <input
              type="text"
              autoFocus
              placeholder="Group Name"
              value={newGroupName}
              onChange={(e) => setNewGroupName(e.target.value)}
              className="w-full bg-background border border-border rounded-lg py-2 px-3 text-sm text-foreground mb-4"
            />
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setCreateGroupOpen(false)}
                className="px-4 py-2 text-xs font-semibold rounded-lg bg-secondary text-foreground hover:bg-secondary/80"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateGroup}
                className="px-4 py-2 text-xs font-semibold rounded-lg bg-primary text-primary-foreground hover:bg-primary/90"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Add Member Modal */}
      {addMemberOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-[#1e1e1e] border border-border rounded-xl p-6 w-96 shadow-2xl animate-in fade-in zoom-in-95">
            <h3 className="text-sm font-bold text-foreground mb-4 flex items-center gap-2">
              <UserPlus size={16} /> Add Group Member
            </h3>

            <div className="space-y-4">
              <div>
                <label className="text-xs text-muted-foreground block mb-1">Display Name</label>
                <input
                  type="text"
                  placeholder="e.g. Claude, GPT-4, Researcher"
                  value={newMemberName}
                  onChange={(e) => setNewMemberName(e.target.value)}
                  className="w-full bg-background border border-border rounded-lg py-2 px-3 text-sm text-foreground"
                />
              </div>

              <div className="relative">
                <label className="text-xs text-muted-foreground block mb-1">Select Model</label>
                <button
                  onClick={() => setModelDropdownOpen((v) => !v)}
                  className="w-full text-left bg-background border border-border rounded-lg py-2 px-3 text-sm text-foreground flex justify-between items-center"
                >
                  <span className="truncate">
                    {selectedModelInfo ? (
                      selectedModelInfo.category === 'external' ? (
                        `${selectedModelInfo.name} (${selectedModelInfo.family || selectedModelInfo.backend || 'Cloud'})`
                      ) : (
                        `${selectedModelInfo.name}${selectedModelInfo.quantization ? ` · ${selectedModelInfo.quantization}` : ''}`
                      )
                    ) : (
                      <span className="text-muted-foreground">Select a model...</span>
                    )}
                  </span>
                  <ChevronDown size={14} className="opacity-50" />
                </button>
                
                {modelDropdownOpen && (
                  <>
                    <div className="fixed inset-0 z-[60]" onClick={() => setModelDropdownOpen(false)} />
                    <div className="absolute left-0 top-full mt-1 w-full bg-[#1e1e1e] border border-border rounded-xl shadow-2xl p-2 z-[70] animate-in fade-in slide-in-from-top-2">
                      
                      {/* Local Models Section */}
                      {localModels.length > 0 && (
                        <>
                          <div className="px-2 py-1.5 text-[10px] font-bold text-muted-foreground uppercase tracking-wider bg-secondary/30 rounded-t-lg">
                            Local Inference
                          </div>
                          <div className="max-h-[25vh] overflow-y-auto">
                            {localModels.map((m) => {
                              const isActive = m.id === newMemberModelId;
                              return (
                                <button
                                  key={m.id}
                                  onClick={() => {
                                    setNewMemberModelId(m.id);
                                    setModelDropdownOpen(false);
                                  }}
                                  className={cn(
                                    "w-full text-left px-3 py-2 rounded-lg text-xs transition-all flex items-center justify-between gap-2 my-0.5",
                                    isActive
                                      ? "bg-primary/10 text-primary font-bold"
                                      : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                                  )}
                                >
                                  <div className="min-w-0">
                                    <div className="font-semibold truncate">
                                      {m.name}{m.quantization ? ` · ${m.quantization}` : ''}
                                    </div>
                                    <div className="text-[10px] opacity-60">{m.size || 'Unknown size'}</div>
                                  </div>
                                  {m.status === "loaded" && (
                                    <span className="text-[9px] px-1.5 py-0.5 rounded-md bg-green-500/10 text-green-500 font-bold shrink-0">LOADED</span>
                                  )}
                                </button>
                              );
                            })}
                          </div>
                        </>
                      )}

                      {/* External Provider Models Section */}
                      {cloudModels.length > 0 && (
                        <>
                          <div className="px-2 py-1.5 text-[10px] font-bold text-muted-foreground uppercase tracking-wider bg-secondary/30 border-t border-border/50">
                            External Providers
                          </div>
                          <div className="max-h-[25vh] overflow-y-auto pt-1">
                            {cloudModels.map((m) => {
                              const isActive = m.id === newMemberModelId;
                              // Approximate provider color using the architecture or provider name
                              const pColor = PROVIDER_COLORS[m.architecture?.toLowerCase() as string] || "#6b7280";
                              return (
                                <button
                                  key={m.id}
                                  onClick={() => {
                                    setNewMemberModelId(m.id);
                                    setModelDropdownOpen(false);
                                  }}
                                  className={cn(
                                    "w-full text-left px-3 py-2 rounded-lg text-xs transition-all flex items-center justify-between gap-2 my-0.5",
                                    isActive
                                      ? "bg-primary/10 text-primary font-bold"
                                      : "text-muted-foreground hover:bg-secondary hover:text-foreground"
                                  )}
                                >
                                  <div className="min-w-0">
                                    <div className="font-semibold truncate">{m.name}</div>
                                    <div className="flex items-center gap-1.5">
                                      <div className="w-1.5 h-1.5 rounded-full" style={{ background: pColor }} />
                                      <div className="text-[10px] opacity-60 truncate">
                                        {m.family || m.architecture || 'API'}
                                      </div>
                                    </div>
                                  </div>
                                  {isActive && (
                                    <span className="text-[9px] px-1.5 py-0.5 rounded-md bg-primary/10 text-primary font-bold shrink-0">SELECTED</span>
                                  )}
                                </button>
                              );
                            })}
                          </div>
                        </>
                      )}
                    </div>
                  </>
                )}
              </div>

              <div>
                <label className="text-xs text-muted-foreground block mb-1">Role/Persona (Optional)</label>
                <input
                  type="text"
                  placeholder="e.g. You are a senior data scientist..."
                  value={newMemberRole}
                  onChange={(e) => setNewMemberRole(e.target.value)}
                  className="w-full bg-background border border-border rounded-lg py-2 px-3 text-sm text-foreground"
                />
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="is_mod"
                  checked={newMemberModerator}
                  onChange={(e) => setNewMemberModerator(e.target.checked)}
                  className="accent-primary rounded"
                />
                <label htmlFor="is_mod" className="text-xs text-foreground flex items-center gap-1 cursor-pointer">
                  Designate as Moderator <Crown size={12} className="text-amber-500" />
                </label>
              </div>
            </div>

            <div className="flex justify-end gap-2 mt-6">
              <button
                onClick={() => setAddMemberOpen(false)}
                className="px-4 py-2 text-xs font-semibold rounded-lg bg-secondary text-foreground hover:bg-secondary/80"
              >
                Cancel
              </button>
              <button
                onClick={handleAddMember}
                disabled={!newMemberName || !newMemberModelId}
                className="px-4 py-2 text-xs font-semibold rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              >
                Add Member
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
