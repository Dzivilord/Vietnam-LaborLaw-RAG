/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow HMR when the dev server is opened from another device on the LAN.
  allowedDevOrigins: ['192.168.100.37'],
};

module.exports = nextConfig;
