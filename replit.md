# Professional AI Trading Bot - Institutional Grade

## Overview

This is a professional-grade, fully autonomous AI trading bot for cryptocurrencies with institutional-level features. The system includes real exchange integrations (Binance, Coinbase), advanced machine learning models, comprehensive risk management, sentiment analysis, on-chain analysis, cross-exchange arbitrage, and a sophisticated React frontend. Built with Node.js/Express backend, PostgreSQL database, and deployed on a single URL.

## User Preferences

Preferred communication style: Simple, everyday language.
Target: 10/10 institutional-grade intelligence and professionalism.

## System Architecture

### Frontend Architecture
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite for fast development and optimized production builds
- **Routing**: Wouter for lightweight client-side routing
- **UI Framework**: shadcn/ui components built on Radix UI primitives
- **Styling**: Tailwind CSS with CSS custom properties for theming
- **State Management**: TanStack Query (React Query) for server state management
- **Canvas Rendering**: Custom HTML5 Canvas implementations for real-time charts

### Backend Architecture
- **Runtime**: Node.js with Express.js framework
- **Language**: TypeScript with ES modules
- **API Pattern**: RESTful API endpoints with JSON responses
- **Database ORM**: Drizzle ORM for type-safe database operations
- **Session Management**: Express sessions with PostgreSQL storage via connect-pg-simple

### Development Setup
- **Monorepo Structure**: Shared TypeScript types and schemas between client and server
- **Development Server**: Vite dev server with HMR integrated with Express
- **Build Process**: Separate client (Vite) and server (esbuild) build pipelines

## Key Components

### Database Schema (shared/schema.ts)
- **Users**: Authentication and user management
- **Portfolio**: Real-time portfolio values and daily changes
- **Positions**: Active trading positions with P&L tracking
- **Trades**: Historical trade execution records
- **Bot Status**: AI bot configuration and operational state
- **Market Data**: Real-time cryptocurrency market information
- **Risk Metrics**: Portfolio risk assessment and metrics
- **Alerts**: System notifications and trading alerts

### Advanced Trading Components
- **Risk Management**: VaR, Monte Carlo simulations, portfolio heat analysis
- **ML Models**: LSTM, Random Forest, Neural Networks, Reinforcement Learning
- **Sentiment Analysis**: News, social media, Fear & Greed Index
- **On-Chain Analysis**: Whale movements, exchange flows, network metrics
- **Arbitrage Engine**: Cross-exchange, statistical, triangular arbitrage
- **Signal Generator**: Multi-factor analysis combining all components

### Frontend Components
- **Dashboard Layout**: Fixed sidebar navigation with main content area
- **Real-time Charts**: Canvas-based price charts and performance visualization
- **Trading Controls**: AI strategy configuration and bot management
- **Data Tables**: Trade history, positions, and order book displays
- **Alert System**: Toast notifications for trading events

### Backend Services
- **Storage Layer**: Abstract storage interface with PostgreSQL implementation
- **API Routes**: RESTful endpoints for all trading data operations
- **Real-time Updates**: Polling-based data refresh for live updates
- **Error Handling**: Centralized error handling with structured responses

## Data Flow

1. **Client Requests**: React components use TanStack Query to fetch data from API endpoints
2. **API Layer**: Express routes validate requests and delegate to storage layer
3. **Database Operations**: Drizzle ORM executes type-safe SQL queries against PostgreSQL
4. **Real-time Updates**: Client polls endpoints at different intervals (3-30 seconds) for live data
5. **State Management**: TanStack Query caches responses and manages loading/error states
6. **UI Updates**: React components automatically re-render when query data changes

## External Dependencies

### Core Runtime
- **Database**: PostgreSQL with Neon serverless connector
- **Authentication**: Session-based with PostgreSQL session store
- **UI Components**: Radix UI primitives for accessibility
- **Form Handling**: React Hook Form with Zod validation

### Development Tools
- **Type Safety**: TypeScript with strict configuration
- **Code Quality**: Path aliases for clean imports
- **Build Optimization**: Vite plugins for development experience
- **Replit Integration**: Cartographer plugin and runtime error overlay

## Deployment Strategy

### Production Build
- **Client**: Vite builds optimized static assets to `dist/public`
- **Server**: esbuild bundles Node.js application to `dist/index.js`
- **Static Serving**: Express serves client assets in production mode
- **Database Migrations**: Drizzle Kit manages schema migrations

### Environment Configuration
- **Database URL**: Required environment variable for PostgreSQL connection
- **Development Mode**: Vite dev server with Express middleware integration
- **Production Mode**: Express serves pre-built static assets

### Key Features
- **Real-time Trading Data**: Live market data and portfolio updates
- **AI Bot Management**: Strategy configuration and operational controls
- **Risk Monitoring**: Portfolio heat maps and drawdown tracking
- **Trade Execution**: Historical trade analysis and performance metrics
- **Alert System**: Configurable notifications for trading events