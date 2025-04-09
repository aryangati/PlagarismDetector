import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructType, StructField
from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer, StringIndexerModel
)
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml import Pipeline, PipelineModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing Spark Session...")
spark = SparkSession.builder.appName('nlpProjectApi').getOrCreate()
spark.sparkContext.setLogLevel("WARN")
logger.info("Spark Session Initialized.")

preprocessing_pipeline_model: PipelineModel = None
nb_model: NaiveBayesModel = None
label_map = {}
fitted_indexer_model: StringIndexerModel = None

def setup_model():
    global preprocessing_pipeline_model, nb_model, label_map, fitted_indexer_model

    logger.info("Starting model setup...")
    data_path = "/user/hadoopuser/bda/SMSSpamCollection.txt"
    logger.info(f"Loading data from: {data_path}")
    schema = StructType([
        StructField("_c0", StringType(), True),
        StructField("_c1", StringType(), True)
    ])
    try:
        data = spark.read.csv(data_path, schema=schema, sep='\t')
        df = data.withColumnRenamed('_c0', 'tag').withColumnRenamed('_c1', 'message')
        if df.count() == 0:
             raise ValueError("Loaded DataFrame is empty. Check data path and format.")
        logger.info("Data loaded successfully.")
        df.printSchema()
        logger.info(f"Total records loaded: {df.count()}")
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}", exc_info=True)
        spark.stop()
        exit(1)

    logger.info("Defining pipeline stages...")
    label_indexer = StringIndexer(inputCol="tag", outputCol="label", handleInvalid="skip")
    regex_tokenizer = RegexTokenizer(inputCol="message", outputCol="words", pattern="\\w+", gaps=False)
    stop_words_remover = StopWordsRemover(inputCol="words", outputCol="removed")
    hashing_tf = HashingTF(inputCol="removed", outputCol="rawFeatures")
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    nb = NaiveBayes(smoothing=1.0, modelType='multinomial', featuresCol='features', labelCol='label')

    logger.info("Fitting StringIndexer...")
    try:
        fitted_indexer_model = label_indexer.fit(df)
        label_map = {i: label for i, label in enumerate(fitted_indexer_model.labels)}
        logger.info(f"StringIndexer fitted. Detected label mapping: {label_map}")
        df_indexed = fitted_indexer_model.transform(df)
    except Exception as e:
        logger.error(f"Error fitting StringIndexer: {e}", exc_info=True)
        spark.stop()
        exit(1)

    preprocessing_pipeline = Pipeline(stages=[regex_tokenizer, stop_words_remover, hashing_tf, idf])
    logger.info("Fitting the preprocessing pipeline...")
    try:
        preprocessing_pipeline_model = preprocessing_pipeline.fit(df_indexed)
        df_featurized = preprocessing_pipeline_model.transform(df_indexed)
        logger.info("Preprocessing pipeline fitted successfully.")
    except Exception as e:
        logger.error(f"Error fitting preprocessing pipeline: {e}", exc_info=True)
        spark.stop()
        exit(1)

    logger.info("Training Naive Bayes model...")
    try:
        nb_model = nb.fit(df_featurized)
        logger.info("Naive Bayes model trained successfully.")
    except Exception as e:
        logger.error(f"Error training Naive Bayes model: {e}", exc_info=True)
        spark.stop()
        exit(1)

    logger.info("Model setup complete.")

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"],   
)

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI startup: Initializing model...")
    setup_model()
    logger.info("FastAPI startup: Model ready.")

@app.get("/predict")
async def predict_spam_api(sms: str):
    if not preprocessing_pipeline_model or not nb_model or not label_map:
        logger.error("Prediction requested before model initialization is complete.")
        return {"error": "Model not ready"}, 503

    logger.info(f"Received prediction request for SMS: '{sms}'")
    input_data = spark.createDataFrame([(sms,)], ["message"])

    try:
        logger.info("Applying preprocessing pipeline...")
        featurized_input = preprocessing_pipeline_model.transform(input_data)
        logger.info("Preprocessing complete.")

        logger.info("Applying Naive Bayes model...")
        prediction_result = nb_model.transform(featurized_input)
        logger.info("Prediction complete.")

        result_row = prediction_result.select("prediction").first()

        if result_row:
            prediction_index = result_row['prediction']
            predicted_label = label_map.get(prediction_index, "unknown")
            logger.info(f"Prediction index: {prediction_index}, Mapped label: {predicted_label}")
            is_spam = (predicted_label == 'spam')
            display_label = "Not Spam" if predicted_label == "ham" else predicted_label.capitalize()
            return {"sms": sms, "predicted_label": display_label, "is_spam": is_spam}
        else:
            logger.warning("Prediction resulted in no output row.")
            return {"error": "Prediction failed to produce a result"}, 500

    except Exception as e:
        logger.error(f"Error during prediction processing: {e}", exc_info=True)
        spark_error_msg = str(e)
        return {"error": f"Prediction processing error: {spark_error_msg}"}, 500


@app.get("/")
async def read_root():
    return {"message": "SMS Spam Detection API. Use /predict?sms=... endpoint."}

if __name__ == "__main__":
    logger.info("Starting Uvicorn server on http://0.0.0.0:5001")
    uvicorn.run(app, host="0.0.0.0", port=5001, reload=False)

    try:
        logger.info("Shutting down Spark Session...")
        spark.stop()
    except Exception as e:
        logger.warning(f"Error stopping Spark session: {e}")