const mongoose = require("mongoose");

const CropSchema = new mongoose.Schema({
    N: Number,
    P: Number,
    K: Number,
    temperature: Number,
    humidity: Number,
    pH: Number,
    rainfall: Number,
    location: String,
    recommended_crop: String,
    timestamp: { type: Date, default: Date.now }
});

module.exports = mongoose.model("Crop", CropSchema);
