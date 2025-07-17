// Initialize Socket.IO connection
const socket = io();

// Global variables
let liveChart;
let tradingActive = false;
let marketData = {};
let signals = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    setupSocketListeners();
    initializeChart();
    
    // Start data refresh intervals
    setInterval(updateDashboard, 5000);
    setInterval(updateCrypto, 30000);
    set
    