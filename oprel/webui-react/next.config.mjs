/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  basePath: '/gui',
  trailingSlash: true,
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  experimental: {
    cpus: 1,
  },
}

export default nextConfig
