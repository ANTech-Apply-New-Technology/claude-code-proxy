FROM node:22-alpine

WORKDIR /app
COPY package.json ./
RUN npm install --omit=dev 2>/dev/null || true
COPY . .

# Use docker config and force 0.0.0.0 binding for Railway
RUN cp server/config.docker.txt server/config.txt && \
    sed -i 's/^host=.*/host=0.0.0.0/' server/config.txt

# Token persistence: mount a volume at /data
ENV HOME=/data
ENV NODE_ENV=production

EXPOSE 42069

CMD ["node", "server/server.js"]
