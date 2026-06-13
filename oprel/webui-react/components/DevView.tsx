"use client"

import { useState, useEffect, useRef, useMemo } from "react"
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from "recharts"
import { Cpu, HardDrive, Zap, Activity, MemoryStick, Layers } from "lucide-react"
import { cn } from "@/services/utils"
import { useApp } from "@/services/context"
import { API } from "@/services/api"

function generateHistory(base: number, variance: number, len = 20) {
  return Array.from({ length: len }, (_, i) => ({
    t: i,
    v: Math.max(0, Math.min(100, base + (Math.random() - 0.5) * variance * 2)),
  }))
}

const COLOR_MAP: Record<string, string> = {
  "text-primary": "#ee4647",
  "text-amber-400": "#f59e0b",
  "text-green-400": "#22c55e",
  "text-blue-400": "#3b82f6",
  "text-purple-400": "#a855f7",
}

function MetricCard({
  icon,
  label,
  value,
  sub,
  color,
  data,
  unit = "%",
  accent,
}: {
  icon: React.ReactNode
  label: string
  value: string
  sub?: string
  color: string
  data: { t: number; v: number }[]
  unit?: string
  accent?: React.ReactNode
}) {
  return (
    <div className="relative bg-[#1a1a1a] border border-border rounded-xl p-5 overflow-hidden flex flex-col h-40">
      {/* Accent icon top right */}
      {accent && (
        <div className="absolute top-3 right-3">
          {accent}
        </div>
      )}

      <div className="flex items-center gap-1.5 text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-1 z-10 relative">
        <span className={cn("shrink-0", color)}>{icon}</span>
        {label}
      </div>

      <div className="text-3xl font-bold text-foreground tracking-tight z-10 relative leading-none mb-1">
        {value}
      </div>

      {sub && (
        <div className="text-[10px] text-muted-foreground z-10 relative mb-auto">{sub}</div>
      )}

      {/* Sparkline */}
      <div className="absolute bottom-0 left-0 right-0 h-16 opacity-15">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
            <Area
              type="monotone"
              dataKey="v"
              stroke={COLOR_MAP[color] ?? "#ee4647"}
              strokeWidth={1.5}
              fill="none"
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

const CUSTOM_TOOLTIP_STYLE = {
  contentStyle: {
    background: "#1e1e1e",
    border: "1px solid rgba(255,255,255,0.06)",
    borderRadius: "8px",
    fontSize: "11px",
    color: "#e5e5e5",
  },
  cursor: { fill: "rgba(255,255,255,0.03)" },
}

function ActiveModelCard() {
  const { models, localModels, activeModelId } = useApp()
  // Match composite ID (repo_id::QUANT) for local models first
  const model = localModels.find((m) => m.id === activeModelId) || 
                models.find((m) => m.id === activeModelId)

  if (!model) {
    return (
      <div className="bg-[#1a1a1a] border border-border rounded-xl p-5 text-center py-10">
        <Layers size={24} className="text-muted-foreground/20 mx-auto mb-2" />
        <p className="text-xs text-muted-foreground">No active model loaded</p>
      </div>
    )
  }

  return (
    <div className="bg-[#1a1a1a] border border-border rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs font-bold text-foreground">Active Model</h3>
        <span className="flex items-center gap-1.5 text-[10px] text-green-400 font-bold">
          <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" /> {model.status.toUpperCase()}
        </span>
      </div>

      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center">
          <Layers size={18} className="text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-bold text-foreground truncate">{model.name}</div>
          <div className="text-[10px] text-muted-foreground truncate">
            {model.parameters !== "Unknown" ? model.parameters : model.family} · {model.quantization}
          </div>
        </div>
      </div>

    </div>
  )
}

export function DevView() {
  const { activeModelId, models } = useApp()
  const [metrics, setMetrics] = useState<any>(null)
  const [analytics, setAnalytics] = useState<any>({ models: [], timeline: [] })
  const [days, setDays] = useState(1) // Default to 24h

  const [cpuData, setCpuData] = useState(() => generateHistory(0, 0))
  const [gpuData, setGpuData] = useState(() => generateHistory(0, 0))
  const [vramData, setVramData] = useState(() => generateHistory(0, 0))
  const [speedData, setSpeedData] = useState(() => generateHistory(0, 0))
  const [ramData, setRamData] = useState(() => generateHistory(0, 0))

  const [cpuVal, setCpuVal] = useState(0)
  const [gpuVal, setGpuVal] = useState(0)
  const [vramVal, setVramVal] = useState(0)
  const [vramTotal, setVramTotal] = useState(0)
  const [speedVal, setSpeedVal] = useState(0)
  const [ramVal, setRamVal] = useState(0)
  const [ramTotal, setRamTotal] = useState(0)

  const updateAnalytics = async (d: number) => {
    try {
      const data = await API.fetchAnalyticsSummary(d)
      setAnalytics(data)
    } catch (err) {
      console.error("Failed to fetch analytics:", err)
    }
  }

  useEffect(() => {
    const updateMetrics = async () => {
      try {
        const data = await API.getMetrics()
        setMetrics(data)

        setCpuVal(data.cpu_usage)
        setGpuVal(data.gpu_usage || 0)
        setVramVal(((data.vram_used_mb as number) ?? 0) / 1024)
        setVramTotal((data.vram_total_mb || 0) / 1024)
        setSpeedVal(data.generation_speed || 0)
        setRamVal(data.ram_used_gb)
        setRamTotal(data.ram_total_gb)

        const push = (arr: { t: number; v: number }[], v: number) =>
          [...arr.slice(1), { t: arr[arr.length - 1].t + 1, v }]

        setCpuData((d) => push(d, data.cpu_usage))
        setGpuData((d) => push(d, data.gpu_usage || 0))
        setVramData((d) => push(d, (data.vram_total_mb && data.vram_used_mb) ? (data.vram_used_mb / data.vram_total_mb * 100) : 0))
        setSpeedData((d) => push(d, data.generation_speed ? Math.min(data.generation_speed * 2, 100) : 0))
        setRamData((d) => push(d, (data.ram_used_gb / data.ram_total_gb) * 100))
      } catch (error) {
        console.error("Failed to fetch metrics:", error)
      }
    }

    updateMetrics()
    updateAnalytics(days)
    const interval = setInterval(() => {
        updateMetrics();
        // Analytics update less frequent
        if (Math.random() > 0.8) updateAnalytics(days);
    }, 2000)
    return () => clearInterval(interval)
  }, [days])

  // Process timeline data for Chart
  const timelineData = useMemo(() => {
    return analytics.timeline.map((t: any) => ({
      hour: days > 1 
        ? new Date(t.hour).toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit' })
        : new Date(t.hour).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      tokens: t.total_tokens,
      tps: t.tps
    }))
  }, [analytics.timeline, days])

  const totalTokens = analytics.models.reduce((acc: number, m: any) => acc + (m.total_prompt_tokens + m.total_completion_tokens), 0);

  return (
    <div className="flex flex-col h-full bg-[#0a0a0a] overflow-y-auto animate-fade-in-up">
      <div className="p-6 max-w-[1400px] w-full mx-auto">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xs font-bold text-muted-foreground uppercase tracking-widest">System & Model Analytics</h2>
          </div>
          <div className="flex items-center gap-4">
            {/* Time range selector */}
            <div className="flex items-center gap-1 bg-secondary/50 p-1 rounded-lg border border-border/50">
              {[
                { label: '24h', val: 1 },
                { label: '7d', val: 7 },
                { label: '30d', val: 30 },
              ].map(r => (
                <button
                  key={r.val}
                  onClick={() => setDays(r.val)}
                  className={cn(
                    "px-3 py-1 text-[10px] font-bold rounded-md transition-all",
                    days === r.val ? "bg-primary text-white shadow-lg" : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  {r.label}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2 text-[10px] text-muted-foreground shrink-0">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
              Live Analytics Active
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3 mb-6">
          <MetricCard
            icon={<Cpu size={12} />}
            label="CPU Usage"
            value={`${cpuVal.toFixed(0)}%`}
            sub="System Process"
            color="text-primary"
            data={cpuData}
          />
          <MetricCard
               icon={<Zap size={12} />}
               label="Current Speed"
               value={`${speedVal.toFixed(1)}`}
               sub="tokens per second"
               color="text-green-400"
               data={speedData}
          />
          <MetricCard
            icon={<HardDrive size={12} />}
            label="VRAM"
            value={`${vramVal.toFixed(1)}`}
            sub={`/ ${vramTotal.toFixed(0)} GB`}
            color="text-amber-400"
            data={vramData}
          />
          <MetricCard
            icon={<MemoryStick size={12} />}
            label="RAM"
            value={`${ramVal.toFixed(1)}`}
            sub={`/ ${ramTotal.toFixed(0)} GB`}
            color="text-green-400"
            data={ramData}
          />
          <div className="bg-[#1a1a1a] border border-border rounded-xl p-5 flex flex-col justify-center">
             <div className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest mb-1">Total Tokens</div>
             <div className="text-3xl font-bold text-foreground">{(totalTokens / 1000).toFixed(1)}k</div>
             <div className="text-[10px] text-muted-foreground mt-1">{days} Day Period</div>
          </div>
        </div>

        <div className="bg-[#1a1a1a] border border-border rounded-xl p-5 ">
          <h3 className="text-xs font-bold text-foreground mb-4">Latency Distribution (ms)</h3>
          <div className="h-[200px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={analytics.models}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
                        <XAxis dataKey="model_id" stroke="#525252" fontSize={10} tickLine={false} axisLine={false} />
                        <YAxis stroke="#525252" fontSize={10} tickLine={false} axisLine={false} tickFormatter={(v) => `${v}ms`} />
                        <Tooltip {...CUSTOM_TOOLTIP_STYLE} />
                        <Bar dataKey="avg_latency" radius={[4, 4, 0, 0]} barSize={40}>
                            {analytics.models.map((m: any, idx: number) => {
                                const knownModel = models.find(km => km.id === m.model_id);
                                const color = knownModel?.category === 'external' ? '#a855f7' : '#ee4647';
                                return <Cell key={`cell-${idx}`} fill={color} />;
                            })}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4 items-start mt-6">
          <div className="lg:col-span-2 bg-[#1a1a1a] border border-border rounded-xl p-5 h-[350px] flex flex-col">
            <div className="flex items-center justify-between mb-6">
                <h3 className="text-xs font-bold text-foreground">Token Volume (Last {days}d)</h3>
            </div>
            
            <div className="flex-1 w-full min-h-[250px]">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={timelineData}>
                        <defs>
                            <linearGradient id="colorTok" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#ee4647" stopOpacity={0.3}/>
                                <stop offset="95%" stopColor="#ee4647" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#262626" vertical={false} />
                        <XAxis dataKey="hour" stroke="#525252" fontSize={10} tickLine={false} axisLine={false} />
                        <YAxis stroke="#525252" fontSize={10} tickLine={false} axisLine={false} tickFormatter={(v) => `${v/1000}k`} />
                        <Tooltip {...CUSTOM_TOOLTIP_STYLE} />
                        <Area type="monotone" dataKey="tokens" stroke="#ee4647" fillOpacity={1} fill="url(#colorTok)" strokeWidth={2} />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
          </div>

          <div className="flex flex-col gap-4">
            <ActiveModelCard />
            <div className="bg-[#1a1a1a] border border-border rounded-xl p-5 flex-1">
                <h3 className="text-xs font-bold text-foreground mb-4">Model Activity Heatmap</h3>
                <div className="space-y-4">
                    {analytics.models.length === 0 ? (
                         <div className="text-center py-8 text-xs text-muted-foreground">No recent activity</div>
                    ) : (
                        analytics.models.map((m: any) => {
                            // Find the color of this model if it exists in our metadata
                            const knownModel = models.find(km => km.id === m.model_id);
                            // Fallback color logic
                            const color = knownModel?.category === 'external' ? '#a855f7' : '#ee4647';
                            
                            return (
                                <div key={m.model_id} className="space-y-1.5">
                                    <div className="flex justify-between items-center text-[10px]">
                                        <div className="flex items-center gap-2 min-w-0">
                                            <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: color }} />
                                            <span className="text-foreground font-bold truncate">{m.model_id}</span>
                                        </div>
                                        <span className="text-muted-foreground shrink-0">{m.total_prompt_tokens + m.total_completion_tokens} tokens</span>
                                    </div>
                                    <div className="h-1.5 w-full bg-secondary rounded-full overflow-hidden">
                                        <div 
                                            className="h-full transition-all duration-500" 
                                            style={{ 
                                                width: `${Math.min(100, (m.total_prompt_tokens + m.total_completion_tokens) / totalTokens * 100)}%`,
                                                backgroundColor: color 
                                            }}
                                        />
                                    </div>
                                </div>
                            );
                        })
                    )}
                </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
