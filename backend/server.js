const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const axios = require("axios");

const app = express();
app.use(express.json());
app.use(cors({
    origin: "http://localhost:3000",
}));

// MongoDB Connection
mongoose.connect("mongodb://localhost:27017/crop", {
    useNewUrlParser: true,
    useUnifiedTopology: true,
})
.then(() => console.log("Connected to MongoDB"))
.catch((err) => console.error("MongoDB Connection Error:", err));

// Define Crop Schema
const Crop = require("./models/crop");

// Weather API Key (Replace with your API key)
const WEATHER_API_KEY = "af04ef4c7d0311f77d1c99e02cd8c4b0";

// Route: Predict Crop Based on Weather
app.post("/predict-crop", async (req, res) => {
    try {
        const { lat, lon, N, P, K, pH } = req.body;

        if (!lat || !lon || N === undefined || P === undefined || K === undefined || pH === undefined) {
            return res.status(400).json({ error: "Missing required fields in request body" });
        }

        // Fetch weather data
        const weatherURL = `http://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&appid=${WEATHER_API_KEY}&units=metric`;
        const weatherResponse = await axios.get(weatherURL);

        if (!weatherResponse.data || !weatherResponse.data.main) {
            return res.status(500).json({ error: "Failed to fetch weather data" });
        }

        const { temp, humidity } = weatherResponse.data.main;
        const rainfall = Math.random() * 100; // Simulated rainfall

        // Prepare data for ML Model
        const predictionData = { N, P, K, Temperature: temp, Humidity: humidity, pH, Rainfall: rainfall };

        // Call ML Model
        const mlResponse = await axios.post("http://127.0.0.1:5003/predict-crop", predictionData);

        if (!mlResponse.data || !mlResponse.data.crop) {
            return res.status(500).json({ error: "Failed to get a prediction from the ML model" });
        }

        const recommended_crop = mlResponse.data.crop;

        // Save to MongoDB
        const newCrop = new Crop({ ...predictionData, location: `${lat},${lon}`, recommended_crop });
        await newCrop.save();

        return res.json({ recommended_crop, weather: { temp, humidity, rainfall } });

    } catch (error) {
        console.error("Server Error:", error.message);
        res.status(500).json({ error: "Prediction failed", details: error.message });
    }
});

// Start Server
const PORT = 5001;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
