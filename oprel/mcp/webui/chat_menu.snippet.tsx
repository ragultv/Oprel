// The Connectors entry of the chat '+' menu, with its flyout (add / manage / per-connector switches).
// Replace the 'Connectors - Soon' button in ChatView with this block.
// Needs lucide icons: Plug, TriangleAlert, Briefcase, Plus, ChevronRight; the Switch component; and chat_state.snippet.tsx

                        <div className="relative">
                          <button
                            onClick={() => setMoreToolsOpen((v) => !v)}
                            aria-expanded={moreToolsOpen}
                            aria-haspopup="menu"
                            className={cn(
                              "w-full text-left px-3 py-2 rounded-xl text-xs transition-all flex items-center gap-2.5 cursor-pointer font-medium",
                              moreToolsOpen ? "bg-secondary text-foreground" : "text-foreground/80 hover:bg-secondary hover:text-foreground"
                            )}
                          >
                            <Plug size={14} className="shrink-0 opacity-80" />
                            <span className="flex-1">Connectors</span>
                            {connectorIssues > 0 && (
                              <span
                                className="flex items-center gap-1 rounded-full bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-400"
                                title={`${connectorIssues} connector${connectorIssues > 1 ? "s need" : " needs"} attention`}
                              >
                                <TriangleAlert size={10} />
                                {connectorIssues}
                              </span>
                            )}
                            <ChevronRight size={12} className="opacity-40" />
                          </button>

                          {moreToolsOpen && (
                            <div
                              role="menu"
                              aria-label="Connectors"
                              className="absolute bottom-0 left-full ml-2 w-64 rounded-2xl border border-border/80 bg-[#1a1a1a]/95 p-1.5 shadow-2xl backdrop-blur-md z-50 flex flex-col gap-0.5"
                            >
                              <button
                                role="menuitem"
                                onClick={() => {
                                  setToolsMenuOpen(false);
                                  router.push("/settings?tab=connectors");
                                }}
                                className="flex items-center gap-2.5 rounded-xl px-3 py-2 text-left text-xs font-medium text-foreground/80 hover:bg-secondary hover:text-foreground"
                              >
                                <Plus size={14} className="shrink-0 opacity-80" />
                                Add connector
                              </button>
                              <button
                                role="menuitem"
                                onClick={() => {
                                  setToolsMenuOpen(false);
                                  router.push("/settings?tab=connectors");
                                }}
                                className="flex items-center gap-2.5 rounded-xl px-3 py-2 text-left text-xs font-medium text-foreground/80 hover:bg-secondary hover:text-foreground"
                              >
                                <Briefcase size={14} className="shrink-0 opacity-80" />
                                Manage connectors
                              </button>
                              {chatConnectors.length > 0 && <div className="mx-1.5 my-1 h-px bg-border/40" />}
                              {chatConnectors.map((c) => {
                                const healthy = c.enabled && c.status === "connected";
                                return (
                                  <div key={c.id} className="flex items-center gap-2.5 rounded-xl px-3 py-2 text-xs font-medium">
                                    <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-secondary text-[10px]">
                                      {c.name.charAt(0).toUpperCase()}
                                    </span>
                                    <span className="min-w-0 flex-1 truncate" title={c.network === "none" ? "Works only on this computer" : "Uses the internet"}>
                                      {c.name}
                                    </span>
                                    {healthy ? (
                                      <Switch
                                        checked={c.use_in_chat}
                                        onCheckedChange={(on) => toggleChatConnector(c, on)}
                                        aria-label={`Use ${c.name} in chat`}
                                      />
                                    ) : (
                                      <span
                                        className="text-amber-400"
                                        title={c.enabled ? `Not running (${c.status}). Open Manage connectors.` : "Turned off in settings"}
                                      >
                                        <TriangleAlert size={14} />
                                      </span>
                                    )}
                                  </div>
                                );
                              })}
                              {chatConnectors.length === 0 && (
                                <p className="px-3 py-2 text-[11px] text-muted-foreground">No connectors yet.</p>
                              )}
                            </div>
                          )}
                        </div>
