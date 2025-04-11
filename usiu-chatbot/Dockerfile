# usiu-chatbot/Dockerfile
FROM node:18.18.2

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

RUN npm run build
RUN npm install -g serve
CMD ["serve", "-s", "build", "-l", "3000"]