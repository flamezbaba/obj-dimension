const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const cocoSsd = require("@tensorflow-models/coco-ssd");
const fs = require("fs");
const path = require("path");

// Set up Express
const app = express();
const PORT = process.env.PORT || 3000;

// Set up Multer for file upload handling
const upload = multer({ dest: "uploads/" });

app.get("/", (req, res) => {
  res.json("it is me");
});

// API route to handle POST requests with image uploads
app.post("/detect", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).send("No image file uploaded.");
    }

    const imagePath = path.join(__dirname, req.file.path);

    // Load and detect objects in the image
    const results = await detectObjects(imagePath);

    // Optionally return the results as JSON
    // If you want to return the modified image with bounding boxes, return a file stream
    return res.json(results);
  } catch (error) {
    console.error("Error during object detection:", error);
    return res.status(500).send("Error during object detection.");
  }
});

// Load the image and perform object detection
async function detectObjects(imagePath) {
  try {
    // Load the COCO-SSD model
    const model = await cocoSsd.load();

    // Convert the image to a tensor
    const imageTensor = tf.node.decodeImage(fs.readFileSync(imagePath));

    // Perform object detection
    const predictions = await model.detect(imageTensor);
    let objects = [];

    // Log and draw bounding boxes for each prediction
    predictions.forEach((prediction) => {
      const [x, y, width, height] = prediction.bbox;
      const area = width * height; // Calculate object area
      let sizeCategory = "";

      // Categorize based on area
      if (area < 20000) {
        sizeCategory = "small";
      } else if (area >= 20000 && area <= 100000) {
        sizeCategory = "medium";
      } else {
        sizeCategory = "large";
      }

      objects.push({
        width: width,
        height: height,
        size: sizeCategory,
        prediction: prediction.class
      });

      console.log(`Object: ${prediction.class} | 
                         Confidence: ${prediction.score.toFixed(2)} | 
                         Bounding Box: [x: ${x}, y: ${y}, width: ${width}, height: ${height}]`);
    });

    if(objects.length > 0){
      return objects[0];
    }

    return null;
  } catch (error) {
    console.error("Error during object detection:", error);
    throw error;
  }
}

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
