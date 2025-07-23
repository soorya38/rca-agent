# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies (use npm ci without --only=production for build dependencies)
RUN npm ci

# Copy source code (excluding node_modules to avoid platform conflicts)
COPY src/ ./src/
COPY public/ ./public/
COPY index.html ./
COPY vite.config.js ./
COPY postcss.config.js ./

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built assets from build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"] 