import type { Metadata, Viewport } from 'next'
import localFont from 'next/font/local'
import './globals.css'
import { AppProvider } from '@/services/context'
import { DownloadProvider } from '@/services/downloadContext'
import { Toaster } from '@/components/ui/toaster'
import { DownloadDialog } from '@/components/DownloadDialog'
import { AppShell } from '@/components/AppShell'

const spaceGrotesk = localFont({
  src: [
    {
      path: '../public/fonts/space-grotesk/space-grotesk-latin.woff2',
      weight: '300 700',
      style: 'normal',
    },
  ],
  variable: '--font-space-grotesk',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Oprel Studio',
  description: 'Local AI model runner — chat, manage and analyze LLMs',
}

export const viewport: Viewport = {
  themeColor: '#0f0f0f',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={spaceGrotesk.variable}>
      <body className="font-sans antialiased bg-background text-foreground h-full overflow-hidden">
        <AppProvider>
          <DownloadProvider>
            <AppShell>{children}</AppShell>
            <DownloadDialog />
            <Toaster />
          </DownloadProvider>
        </AppProvider>
      </body>
    </html>
  )
}
