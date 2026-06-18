"use client"

import { useState, useEffect, useRef } from "react"
import { Send, Users, Cpu, Cloud, Crown, Loader2, Bot, Brain, ChevronDown } from "lucide-react"
import { GroupsAPI, type Group, type GroupMessage, type GroupMember, API_BASE } from "@/services/api"
import { cn } from "@/services/utils"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import remarkBreaks from "remark-breaks"

function getInitials(name: string) {
  return name.substring(0, 2).toUpperCase()
}

export function GroupChatView({ groupId }: { groupId: string }) {
  const [group, setGroup] = useState<Group | null>(null)
  const [messages, setMessages] = useState<GroupMessage[]>([])
  // msgId -> [{memberId, emoji}]
  const [reactions, setReactions] = useState<Record<string, { memberId: string; emoji: string }[]>>({})
  const [input, setInput] = useState("")
  const [roundState, setRoundState] = useState<string | null>(null)
  
  const [mentionPopupVisible, setMentionPopupVisible] = useState(false)
  const [mentionCursorPos, setMentionCursorPos] = useState({ top: 0, left: 0 })
  const [mentionSearch, setMentionSearch] = useState("")
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    GroupsAPI.getGroup(groupId).then(setGroup).catch(console.error)
    GroupsAPI.fetchMessages(groupId).then((msgs) => {
      // Sort by created_at for WhatsApp style ordering
      setMessages(msgs.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()))
    }).catch(console.error)
    
    // Connect WS
    const wsUrl = new URL(`${API_BASE}/groups/${groupId}/ws`, window.location.href)
    wsUrl.protocol = wsUrl.protocol.replace('http', 'ws')
    
    const ws = new WebSocket(wsUrl.toString())
    wsRef.current = ws
    
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data)
        if (data.type === "state_change") {
          setRoundState(data.state)
          if (data.state === "done") {
             // Refresh messages to catch anything missed
             GroupsAPI.fetchMessages(groupId).then((msgs) => {
               setMessages(msgs.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()))
             })
             setRoundState(null)
          }
        } else if (data.type === "reaction_added") {
          setReactions(prev => {
            const existing = prev[data.message_id] || []
            return {
              ...prev,
              [data.message_id]: [...existing, { memberId: data.member_id, emoji: data.emoji }]
            }
          })
        } else if (data.type === "agent_action") {
          // Just a toast or silent
        } else if (data.type === "message_added") {
          setMessages(prev => [...prev, data.message].sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()))
        }
      } catch (err) {
        console.error("WS Parse error", err)
      }
    }
    
    return () => {
      ws.close()
    }
  }, [groupId])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, reactions, roundState])

  const handleSend = async () => {
    if (!input.trim() || roundState) return
    const text = input
    setInput("")
    setMentionPopupVisible(false)
    
    // Optimistic user message
    const tempMsg: GroupMessage = {
      id: `temp_${Date.now()}`,
      group_id: groupId,
      round_id: "temp",
      sender_type: "user",
      content: text,
      message_type: "trigger",
      sequence_number: 999999,
      created_at: new Date().toISOString()
    }
    setMessages(prev => [...prev, tempMsg])
    
    try {
      await GroupsAPI.postMessage(groupId, text)
      // The WS will handle state changes. The response will trigger round generation.
    } catch (e) {
      console.error(e)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      if (!mentionPopupVisible) {
        handleSend()
      } else {
        // Select first mention if popup is visible
        insertMention(filteredMembers[0])
      }
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value
    setInput(val)
    
    // Simple Mention Logic
    const cursor = e.target.selectionStart
    const textBeforeCursor = val.slice(0, cursor)
    const match = textBeforeCursor.match(/@(\w*)$/)
    
    if (match) {
      setMentionSearch(match[1])
      setMentionPopupVisible(true)
      // Simple positioning estimate
      setMentionCursorPos({ top: -120, left: 10 }) 
    } else {
      setMentionPopupVisible(false)
    }
  }

  const insertMention = (member: GroupMember | 'all') => {
    if (!inputRef.current) return
    const val = input
    const cursor = inputRef.current.selectionStart
    const textBeforeCursor = val.slice(0, cursor)
    const textAfterCursor = val.slice(cursor)
    
    const replacement = member === 'all' ? '@all ' : `@${member.display_name} `
    const newTextBefore = textBeforeCursor.replace(/@\w*$/, replacement)
    
    setInput(newTextBefore + textAfterCursor)
    setMentionPopupVisible(false)
    
    setTimeout(() => {
      inputRef.current?.focus()
    }, 0)
  }

  const membersList = group?.members || []
  const filteredMembers = membersList.filter(m => m.display_name.toLowerCase().includes(mentionSearch.toLowerCase()))

  return (
    <div className="flex flex-col h-full bg-[#0a0a0a]">
      {/* Header */}
      <div className="flex-none p-4 border-b border-border bg-[#171717] flex justify-between items-center z-10">
        <div>
          <h2 className="font-bold text-lg text-foreground tracking-tight">{group?.name || "Loading..."}</h2>
          <div className="text-xs text-muted-foreground flex gap-2 items-center">
            <span>{membersList.length} members</span>
            {membersList.length > 0 && (
              <div className="flex -space-x-1 ml-2">
                {membersList.map(m => (
                  <div key={m.id} className="w-5 h-5 rounded-full bg-secondary border border-border flex items-center justify-center text-[8px] font-bold" title={m.display_name}>
                    {getInitials(m.display_name)}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Messages Area - Wider Design (max-w-5xl) */}
      <div className="flex-1 overflow-y-auto p-4 md:p-8">
        <div className="max-w-5xl mx-auto space-y-6">
          {messages.map((msg) => {
            const isUser = msg.sender_type === "user"
            const member = membersList.find(m => m.id === msg.member_id)
            const isInterrupt = msg.message_type === "interrupt"
            const isFinal = msg.message_type === "final_answer"
            
            const msgReactions = reactions[msg.id] || []

            // Aggregate reactions
            const reactionCounts: Record<string, number> = {}
            msgReactions.forEach(r => {
               reactionCounts[r.emoji] = (reactionCounts[r.emoji] || 0) + 1
            })

            return (
              <div key={msg.id} className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}>
                {!isUser && (
                  <div className="mr-3 shrink-0 flex flex-col items-center">
                    <div className={cn(
                      "w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm shadow-sm",
                      isFinal ? "bg-primary text-primary-foreground" :
                      isInterrupt ? "bg-amber-600/20 border border-amber-600/50 text-amber-500" :
                      "bg-secondary border border-border text-foreground"
                    )}>
                      {isFinal ? <Bot size={18} /> : member ? getInitials(member.display_name) : "SYS"}
                    </div>
                  </div>
                )}
                
                <div className={cn(
                  "max-w-[80%] relative",
                  isUser ? "ml-auto" : "mr-auto"
                )}>
                  {/* Sender Name and Time */}
                  <div className="text-xs font-semibold text-muted-foreground mb-1 ml-1 flex items-center justify-between">
                    <div className="flex items-center gap-1.5">
                      {isUser ? "You" : (member?.display_name || "System")}
                      {!isUser && member?.kind === "local" && <Cpu size={10} />}
                      {!isUser && member?.kind === "cloud" && <Cloud size={10} />}
                      {!isUser && member?.is_moderator === 1 && <Crown size={10} className="text-amber-500" />}
                    </div>
                    <span className="text-[10px] opacity-70 font-normal ml-3">
                      {new Date(msg.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                  
                  {/* Message Bubble */}
                  <div className={cn(
                    "p-4 rounded-2xl shadow-sm text-sm whitespace-pre-wrap leading-relaxed relative",
                    isUser ? "bg-primary text-primary-foreground rounded-tr-sm" :
                    isInterrupt ? "bg-[#251f14] border border-amber-600/30 text-amber-50 rounded-tl-sm shadow-inner" :
                    isFinal ? "bg-primary/10 border border-primary/20 text-primary-foreground rounded-tl-sm" :
                    "bg-[#1e1e1e] border border-border text-foreground rounded-tl-sm"
                  )}>
                    {(() => {
                      let cleaned = msg.content || ""
                      let thinking = ""
                      const startIdx = cleaned.indexOf("<think>")
                      if (startIdx !== -1) {
                        const endIdx = cleaned.indexOf("</think>", startIdx + 7)
                        if (endIdx !== -1) {
                          thinking = cleaned.substring(startIdx + 7, endIdx).trim()
                          cleaned = cleaned.substring(0, startIdx).trim() + "\n\n" + cleaned.substring(endIdx + 8).trim()
                        } else {
                          thinking = cleaned.substring(startIdx + 7).trim()
                          cleaned = cleaned.substring(0, startIdx).trim()
                        }
                      }
                      
                      return (
                        <div className="flex flex-col gap-2">
                          {thinking && (
                            <details className="group marker:content-[''] border border-primary/10 rounded-lg bg-primary/5 p-3 text-xs text-primary/70 mb-2 cursor-pointer">
                              <summary className="flex items-center gap-2 font-semibold uppercase tracking-widest outline-none">
                                <Brain size={13} className="transition-transform group-open:rotate-180" />
                                Thinking Process
                              </summary>
                              <div className="mt-2 pt-2 border-t border-primary/10 whitespace-pre-wrap italic opacity-80">
                                {thinking}
                              </div>
                            </details>
                          )}
                          <div className="oprel-response prose prose-invert prose-p:leading-relaxed max-w-none break-words">
                            <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
                              {cleaned}
                            </ReactMarkdown>
                          </div>
                        </div>
                      )
                    })()}

                    {/* Inline Reactions */}
                    {Object.keys(reactionCounts).length > 0 && (
                      <div className="absolute -bottom-3 left-4 flex gap-1 z-10">
                        {Object.entries(reactionCounts).map(([emoji, count]) => (
                          <div key={emoji} className="px-2 py-0.5 rounded-full bg-[#2a2a2a] border border-border shadow-md text-xs flex items-center gap-1 animate-in zoom-in slide-in-from-bottom-2">
                            <span>{emoji}</span>
                            {count > 1 && <span className="font-bold text-[10px]">{count}</span>}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  {/* Spacer for reactions to not overlap next message */}
                  {Object.keys(reactionCounts).length > 0 && <div className="h-4"></div>}
                </div>
              </div>
            )
          })}

          {/* Real-time Status Indicator */}
          {roundState && roundState !== "done" && (
            <div className="flex justify-start">
               <div className="w-10 mr-3"></div>
               <div className="bg-[#1e1e1e]/50 border border-border/50 rounded-lg px-4 py-3 flex items-center gap-3 text-sm text-muted-foreground animate-pulse">
                 <Loader2 size={16} className="animate-spin text-primary" />
                 <span>
                    {roundState === "relevance" ? "Agents analyzing relevance..." :
                     roundState === "generation" ? "Agents typing..." :
                     roundState === "interrupt" ? "Checking for interrupts..." :
                     roundState === "moderation" ? "Moderator analyzing consensus..." :
                     "Processing..."}
                 </span>
               </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="flex-none p-4 bg-[#171717] border-t border-border">
        <div className="max-w-5xl mx-auto relative">
          
          {/* Mention Popup */}
          {mentionPopupVisible && (
            <div 
              className="absolute bg-[#1e1e1e] border border-border rounded-xl shadow-2xl p-2 w-64 z-50 flex flex-col gap-1 max-h-48 overflow-y-auto animate-in slide-in-from-bottom-2 fade-in"
              style={{ bottom: "100%", left: "10px", marginBottom: "10px" }}
            >
              <div className="text-[10px] font-bold text-muted-foreground uppercase px-2 py-1 tracking-wider">Mention</div>
              <button 
                onClick={() => insertMention('all')}
                className="w-full text-left px-3 py-2 rounded-lg text-sm text-foreground hover:bg-secondary flex items-center gap-2"
              >
                <Users size={14} className="text-primary" />
                @all
              </button>
              {filteredMembers.map(m => (
                <button 
                  key={m.id}
                  onClick={() => insertMention(m)}
                  className="w-full text-left px-3 py-2 rounded-lg text-sm text-foreground hover:bg-secondary flex items-center justify-between"
                >
                  <span>@{m.display_name}</span>
                  <div className="flex items-center gap-1 opacity-50">
                    {m.kind === "local" ? <Cpu size={12} /> : <Cloud size={12} />}
                  </div>
                </button>
              ))}
            </div>
          )}

          <div className="relative flex items-end bg-[#1e1e1e] rounded-2xl border border-border shadow-sm focus-within:ring-1 focus-within:ring-primary/50 transition-all">
            <textarea
              ref={inputRef}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              disabled={!!roundState && roundState !== "done"}
              placeholder={
                roundState && roundState !== "done"
                  ? "AI team is currently thinking..."
                  : "Message the group, or @mention an agent..."
              }
              className="w-full bg-transparent resize-none py-4 pl-5 pr-12 text-sm focus:outline-none disabled:opacity-50 min-h-[56px] max-h-48"
              rows={1}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || (!!roundState && roundState !== "done")}
              className="absolute right-3 bottom-3 p-2 rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:hover:bg-primary transition-colors flex items-center justify-center shadow-sm"
            >
              <Send size={16} />
            </button>
          </div>
          <div className="text-center mt-2 text-[10px] text-muted-foreground">
            Oprel AI Groups allows multi-agent reasoning. Use @mentions to direct questions.
          </div>
        </div>
      </div>
    </div>
  )
}
