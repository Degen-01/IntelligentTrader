import { OpenAI } from 'openai';
import { storage } from '../storage-database';
import axios from 'axios';
import Sentiment from 'sentiment';

interface SentimentData {
  symbol: string;
  source: string;
  content: string;
  sentiment: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
  score: number;
  confidence: number;
  timestamp: Date;
  keywords: string[];
  entities: string[];
  influence: number;
}

interface NewsArticle {
  title: string;
  content: string;
  url: string;
  publishedAt: Date;
  source: string;
  sentiment?: SentimentData;
}

interface SocialPost {
  id: string;
  content: string;
  author: string;
  platform: string;
  engagement: number;
  timestamp: Date;
  sentiment?: SentimentData;
}

interface SentimentSignal {
  symbol: string;
  overallSentiment: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
  score: number;
  confidence: number;
  momentum: number;
  sources: {
    news: number;
    social: number;
    onchain: number;
  };
  keyFactors: string[];
  timestamp: Date;
}

export class SentimentAnalysis {
  private openai?: OpenAI;
  private sentiment: Sentiment;
  private newsAPIs: { name: string; url: string; key?: string }[] = [];
  private socialAPIs: { name: string; url: string; key?: string }[] = [];

  constructor(openaiApiKey?: string) {
    if (openaiApiKey) {
      this.openai = new OpenAI({ apiKey: openaiApiKey });
    }
    
    this.sentiment = new Sentiment();
    this.initializeAPIs();
  }

  private initializeAPIs() {
    // News APIs (using free endpoints when possible)
    this.newsAPIs = [
      { name: 'CoinDesk', url: 'https://api.coindesk.com/v1/news.json' },
      { name: 'CryptoNews', url: 'https://cryptonews.com/api/v1/news' },
      { name: 'NewsAPI', url: 'https://newsapi.org/v2/everything', key: process.env.NEWS_API_KEY },
    ];

    // Social APIs
    this.socialAPIs = [
      { name: 'Reddit', url: 'https://www.reddit.com/r/cryptocurrency.json' },
      { name: 'Twitter', url: 'https://api.twitter.com/2/tweets/search/recent', key: process.env.TWITTER_API_KEY },
    ];
  }

  async analyzeSentiment(symbol: string): Promise<SentimentSignal> {
    try {
      // Collect data from multiple sources
      const newsData = await this.collectNewsData(symbol);
      const socialData = await this.collectSocialData(symbol);
      
      // Analyze sentiment for each source
      const newsAnalysis = await this.analyzeNewsData(newsData);
      const socialAnalysis = await this.analyzeSocialData(socialData);
      
      // Combine and weight the signals
      const combinedSignal = this.combineSignals(symbol, newsAnalysis, socialAnalysis);
      
      // Store the analysis
      await this.storeSentimentAnalysis(combinedSignal);
      
      return combinedSignal;
    } catch (error) {
      console.error('Error analyzing sentiment:', error);
      return this.getDefaultSentiment(symbol);
    }
  }

  private async collectNewsData(symbol: string): Promise<NewsArticle[]> {
    const articles: NewsArticle[] = [];
    const keywords = [symbol, symbol.replace('/', ''), 'crypto', 'cryptocurrency', 'bitcoin', 'ethereum'];
    
    for (const api of this.newsAPIs) {
      try {
        let response;
        
        if (api.name === 'CoinDesk') {
          response = await axios.get(api.url);
          const data = response.data;
          
          if (data.articles) {
            for (const article of data.articles.slice(0, 10)) {
              articles.push({
                title: article.title,
                content: article.description || article.title,
                url: article.url,
                publishedAt: new Date(article.publishedAt || Date.now()),
                source: api.name
              });
            }
          }
        } else if (api.name === 'NewsAPI' && api.key) {
          response = await axios.get(api.url, {
            params: {
              q: keywords.join(' OR '),
              apiKey: api.key,
              language: 'en',
              sortBy: 'publishedAt',
              pageSize: 20
            }
          });
          
          const data = response.data;
          if (data.articles) {
            for (const article of data.articles) {
              articles.push({
                title: article.title,
                content: article.description || article.title,
                url: article.url,
                publishedAt: new Date(article.publishedAt),
                source: api.name
              });
            }
          }
        }
      } catch (error) {
        console.error(`Error collecting news from ${api.name}:`, error);
      }
    }
    
    return articles;
  }

  private async collectSocialData(symbol: string): Promise<SocialPost[]> {
    const posts: SocialPost[] = [];
    const keywords = [symbol, symbol.replace('/', ''), 'crypto', 'cryptocurrency'];
    
    for (const api of this.socialAPIs) {
      try {
        let response;
        
        if (api.name === 'Reddit') {
          response = await axios.get(api.url);
          const data = response.data;
          
          if (data.data && data.data.children) {
            for (const post of data.data.children.slice(0, 20)) {
              const postData = post.data;
              
              // Check if post is relevant to our symbol
              const isRelevant = keywords.some(keyword => 
                postData.title.toLowerCase().includes(keyword.toLowerCase()) ||
                postData.selftext.toLowerCase().includes(keyword.toLowerCase())
              );
              
              if (isRelevant) {
                posts.push({
                  id: postData.id,
                  content: `${postData.title} ${postData.selftext}`,
                  author: postData.author,
                  platform: 'Reddit',
                  engagement: postData.score + postData.num_comments,
                  timestamp: new Date(postData.created_utc * 1000)
                });
              }
            }
          }
        }
      } catch (error) {
        console.error(`Error collecting social data from ${api.name}:`, error);
      }
    }
    
    return posts;
  }

  private async analyzeNewsData(articles: NewsArticle[]): Promise<SentimentData[]> {
    const analysis: SentimentData[] = [];
    
    for (const article of articles) {
      try {
        const sentimentData = await this.analyzeSingleText(
          article.content,
          article.title,
          'news',
          article.source
        );
        
        sentimentData.influence = this.calculateNewsInfluence(article);
        article.sentiment = sentimentData;
        analysis.push(sentimentData);
      } catch (error) {
        console.error('Error analyzing news article:', error);
      }
    }
    
    return analysis;
  }

  private async analyzeSocialData(posts: SocialPost[]): Promise<SentimentData[]> {
    const analysis: SentimentData[] = [];
    
    for (const post of posts) {
      try {
        const sentimentData = await this.analyzeSingleText(
          post.content,
          post.platform,
          'social',
          post.author
        );
        
        sentimentData.influence = this.calculateSocialInfluence(post);
        post.sentiment = sentimentData;
        analysis.push(sentimentData);
      } catch (error) {
        console.error('Error analyzing social post:', error);
      }
    }
    
    return analysis;
  }

  private async analyzeSingleText(
    text: string,
    context: string,
    type: 'news' | 'social',
    source: string
  ): Promise<SentimentData> {
    // Basic sentiment analysis using sentiment library
    const basicSentiment = this.sentiment.analyze(text);
    
    // Enhanced analysis with OpenAI if available
    let enhancedSentiment = null;
    if (this.openai) {
      try {
        const response = await this.openai.chat.completions.create({
          model: 'gpt-3.5-turbo',
          messages: [{
            role: 'user',
            content: `Analyze the sentiment of this ${type} text about cryptocurrency: "${text}". 
            Return a JSON object with:
            - sentiment: "POSITIVE", "NEGATIVE", or "NEUTRAL"
            - score: number between -1 and 1
            - confidence: number between 0 and 1
            - keywords: array of important words
            - entities: array of mentioned cryptocurrencies or companies`
          }],
          max_tokens: 300,
          temperature: 0.1
        });
        
        const content = response.choices[0]?.message?.content;
        if (content) {
          enhancedSentiment = JSON.parse(content);
        }
      } catch (error) {
        console.error('Error with OpenAI sentiment analysis:', error);
      }
    }
    
    // Combine results
    const sentiment = enhancedSentiment?.sentiment || 
      (basicSentiment.score > 0 ? 'POSITIVE' : 
       basicSentiment.score < 0 ? 'NEGATIVE' : 'NEUTRAL');
    
    const score = enhancedSentiment?.score || 
      Math.max(-1, Math.min(1, basicSentiment.score / 10));
    
    const confidence = enhancedSentiment?.confidence || 
      Math.min(1, Math.abs(score) + 0.3);
    
    const keywords = enhancedSentiment?.keywords || 
      basicSentiment.positive.concat(basicSentiment.negative);
    
    const entities = enhancedSentiment?.entities || 
      this.extractCryptoEntities(text);
    
    return {
      symbol: context,
      source,
      content: text,
      sentiment,
      score,
      confidence,
      timestamp: new Date(),
      keywords,
      entities,
      influence: 1 // Will be calculated later
    };
  }

  private extractCryptoEntities(text: string): string[] {
    const cryptoTerms = [
      'bitcoin', 'btc', 'ethereum', 'eth', 'binance', 'bnb', 'cardano', 'ada',
      'solana', 'sol', 'polkadot', 'dot', 'chainlink', 'link', 'polygon', 'matic',
      'avalanche', 'avax', 'cosmos', 'atom', 'tron', 'trx', 'litecoin', 'ltc',
      'stellar', 'xlm', 'monero', 'xmr', 'dogecoin', 'doge', 'shiba', 'shib'
    ];
    
    const entities: string[] = [];
    const lowerText = text.toLowerCase();
    
    for (const term of cryptoTerms) {
      if (lowerText.includes(term)) {
        entities.push(term.toUpperCase());
      }
    }
    
    return [...new Set(entities)];
  }

  private calculateNewsInfluence(article: NewsArticle): number {
    let influence = 1;
    
    // Source credibility
    const credibleSources = ['CoinDesk', 'CoinTelegraph', 'Reuters', 'Bloomberg'];
    if (credibleSources.includes(article.source)) {
      influence *= 1.5;
    }
    
    // Recency
    const hoursOld = (Date.now() - article.publishedAt.getTime()) / (1000 * 60 * 60);
    if (hoursOld < 1) influence *= 2;
    else if (hoursOld < 6) influence *= 1.5;
    else if (hoursOld < 24) influence *= 1.2;
    else influence *= 0.8;
    
    return Math.min(3, influence);
  }

  private calculateSocialInfluence(post: SocialPost): number {
    let influence = 1;
    
    // Engagement weight
    if (post.engagement > 100) influence *= 1.5;
    else if (post.engagement > 50) influence *= 1.3;
    else if (post.engagement > 20) influence *= 1.1;
    
    // Platform weight
    if (post.platform === 'Twitter') influence *= 1.2;
    else if (post.platform === 'Reddit') influence *= 1.1;
    
    // Recency
    const hoursOld = (Date.now() - post.timestamp.getTime()) / (1000 * 60 * 60);
    if (hoursOld < 1) influence *= 1.5;
    else if (hoursOld < 6) influence *= 1.2;
    else if (hoursOld > 24) influence *= 0.7;
    
    return Math.min(2, influence);
  }

  private combineSignals(
    symbol: string,
    newsAnalysis: SentimentData[],
    socialAnalysis: SentimentData[]
  ): SentimentSignal {
    const allData = [...newsAnalysis, ...socialAnalysis];
    
    if (allData.length === 0) {
      return this.getDefaultSentiment(symbol);
    }
    
    // Calculate weighted scores
    let totalScore = 0;
    let totalWeight = 0;
    const keyFactors: string[] = [];
    
    for (const data of allData) {
      const weight = data.influence * data.confidence;
      totalScore += data.score * weight;
      totalWeight += weight;
      
      if (data.keywords.length > 0) {
        keyFactors.push(...data.keywords.slice(0, 2));
      }
    }
    
    const averageScore = totalWeight > 0 ? totalScore / totalWeight : 0;
    const confidence = Math.min(1, totalWeight / allData.length);
    
    // Determine overall sentiment
    let overallSentiment: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
    if (averageScore > 0.1) {
      overallSentiment = 'POSITIVE';
    } else if (averageScore < -0.1) {
      overallSentiment = 'NEGATIVE';
    } else {
      overallSentiment = 'NEUTRAL';
    }
    
    // Calculate momentum (change in sentiment over time)
    const recentData = allData.filter(d => 
      Date.now() - d.timestamp.getTime() < 6 * 60 * 60 * 1000 // Last 6 hours
    );
    const olderData = allData.filter(d => 
      Date.now() - d.timestamp.getTime() >= 6 * 60 * 60 * 1000
    );
    
    const recentScore = recentData.length > 0 ? 
      recentData.reduce((sum, d) => sum + d.score, 0) / recentData.length : 0;
    const olderScore = olderData.length > 0 ? 
      olderData.reduce((sum, d) => sum + d.score, 0) / olderData.length : 0;
    
    const momentum = recentScore - olderScore;
    
    // Calculate source breakdown
    const newsScore = newsAnalysis.length > 0 ? 
      newsAnalysis.reduce((sum, d) => sum + d.score, 0) / newsAnalysis.length : 0;
    const socialScore = socialAnalysis.length > 0 ? 
      socialAnalysis.reduce((sum, d) => sum + d.score, 0) / socialAnalysis.length : 0;
    
    return {
      symbol,
      overallSentiment,
      score: averageScore,
      confidence,
      momentum,
      sources: {
        news: newsScore,
        social: socialScore,
        onchain: 0 // Will be filled by on-chain analysis
      },
      keyFactors: [...new Set(keyFactors)].slice(0, 10),
      timestamp: new Date()
    };
  }

  private getDefaultSentiment(symbol: string): SentimentSignal {
    return {
      symbol,
      overallSentiment: 'NEUTRAL',
      score: 0,
      confidence: 0,
      momentum: 0,
      sources: {
        news: 0,
        social: 0,
        onchain: 0
      },
      keyFactors: [],
      timestamp: new Date()
    };
  }

  private async storeSentimentAnalysis(signal: SentimentSignal): Promise<void> {
    try {
      await storage.createAlert({
        userId: 'system',
        type: signal.overallSentiment === 'POSITIVE' ? 'success' : 
              signal.overallSentiment === 'NEGATIVE' ? 'danger' : 'info',
        title: `Sentiment Analysis: ${signal.symbol}`,
        message: `${signal.overallSentiment} sentiment (${(signal.score * 100).toFixed(1)}%) - Key factors: ${signal.keyFactors.slice(0, 3).join(', ')}`
      });
    } catch (error) {
      console.error('Error storing sentiment analysis:', error);
    }
  }

  // Fear & Greed Index calculation
  async calculateFearGreedIndex(symbol: string): Promise<{
    index: number;
    level: 'EXTREME_FEAR' | 'FEAR' | 'NEUTRAL' | 'GREED' | 'EXTREME_GREED';
    components: { [key: string]: number };
  }> {
    try {
      const sentiment = await this.analyzeSentiment(symbol);
      const marketData = await storage.getMarketData(symbol);
      
      const components = {
        sentiment: (sentiment.score + 1) * 50, // Convert to 0-100 scale
        momentum: Math.max(0, Math.min(100, (sentiment.momentum + 1) * 50)),
        volume: 50, // Would need real volume analysis
        social: (sentiment.sources.social + 1) * 50,
        surveys: 50, // Would need survey data
        dominance: 50, // Would need market dominance data
        trends: 50 // Would need Google trends data
      };
      
      // Weight the components
      const weights = {
        sentiment: 0.25,
        momentum: 0.15,
        volume: 0.15,
        social: 0.15,
        surveys: 0.1,
        dominance: 0.1,
        trends: 0.1
      };
      
      const index = Object.entries(components).reduce((sum, [key, value]) => {
        return sum + (value * weights[key as keyof typeof weights]);
      }, 0);
      
      let level: 'EXTREME_FEAR' | 'FEAR' | 'NEUTRAL' | 'GREED' | 'EXTREME_GREED';
      
      if (index <= 20) level = 'EXTREME_FEAR';
      else if (index <= 40) level = 'FEAR';
      else if (index <= 60) level = 'NEUTRAL';
      else if (index <= 80) level = 'GREED';
      else level = 'EXTREME_GREED';
      
      return { index, level, components };
    } catch (error) {
      console.error('Error calculating Fear & Greed Index:', error);
      return {
        index: 50,
        level: 'NEUTRAL',
        components: {}
      };
    }
  }

  // Real-time sentiment monitoring
  async startSentimentMonitoring(symbols: string[], interval: number = 300000): Promise<void> {
    console.log(`Starting sentiment monitoring for ${symbols.join(', ')}`);
    
    const monitor = async () => {
      for (const symbol of symbols) {
        try {
          const sentiment = await this.analyzeSentiment(symbol);
          
          // Check for significant changes
          if (Math.abs(sentiment.score) > 0.7 || Math.abs(sentiment.momentum) > 0.5) {
            await storage.createAlert({
              userId: 'system',
              type: 'warning',
              title: `Significant Sentiment Change: ${symbol}`,
              message: `${sentiment.overallSentiment} sentiment detected with ${(sentiment.confidence * 100).toFixed(1)}% confidence`
            });
          }
        } catch (error) {
          console.error(`Error monitoring sentiment for ${symbol}:`, error);
        }
      }
    };
    
    // Initial analysis
    await monitor();
    
    // Set up periodic monitoring
    setInterval(monitor, interval);
  }

  // Get historical sentiment data
  async getHistoricalSentiment(symbol: string, days: number = 7): Promise<SentimentData[]> {
    // This would typically query a database of historical sentiment data
    // For now, return mock data
    const historical: SentimentData[] = [];
    
    for (let i = days; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      
      historical.push({
        symbol,
        source: 'historical',
        content: `Historical sentiment for ${symbol}`,
        sentiment: Math.random() > 0.5 ? 'POSITIVE' : 'NEGATIVE',
        score: (Math.random() - 0.5) * 2,
        confidence: Math.random() * 0.5 + 0.5,
        timestamp: date,
        keywords: ['crypto', 'trading'],
        entities: [symbol],
        influence: 1
      });
    }
    
    return historical;
  }
}

export default SentimentAnalysis;