using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Spam_Detection.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace Spam_Detection.ML_Model
{
    public class SpamDetectionMLModel
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string TrainDataPath => Path.Combine(AppPath, "..", "..", "..", "Data", "SMSSpamCollection.csv");
        private MLContext mlContext;
        private ITransformer _model;
        private EstimatorChain<TransformerChain<KeyToValueMappingTransformer>> _trainingPipeline;
        private IDataView _data;
        public SpamDetectionMLModel()
        {
            mlContext = null;
            _model = null;
            _trainingPipeline = null;
            _data = null;
        }
        public void Build()
        {
            // Set up the MLContext, which is a catalog of components in ML.NET.
            mlContext = new MLContext();
            // Specify the schema for spam data and read it into DataView.
            _data = mlContext.Data.LoadFromTextFile<SpamInput>(path: TrainDataPath, hasHeader: true, separatorChar: '\t');
            // Data process configuration with pipeline data transformations 
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
                                      .Append(mlContext.Transforms.Text.FeaturizeText("FeaturesText", new Microsoft.ML.Transforms.Text.TextFeaturizingEstimator.Options
                                      {
                                          WordFeatureExtractor = new Microsoft.ML.Transforms.Text.WordBagEstimator.Options { NgramLength = 2, UseAllLengths = true },
                                          CharFeatureExtractor = new Microsoft.ML.Transforms.Text.WordBagEstimator.Options { NgramLength = 3, UseAllLengths = false },
                                      }, "Message"))
                                      .Append(mlContext.Transforms.CopyColumns("Features", "FeaturesText"))
                                      .Append(mlContext.Transforms.NormalizeLpNorm("Features", "Features"))
                                      .AppendCacheCheckpoint(mlContext);
            // Set the training algorithm 
            var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Label", numberOfIterations: 10, featureColumnName: "Features"), labelColumnName: "Label")
                                                  .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));
            _trainingPipeline = dataProcessPipeline.Append(trainer);
        }
        public void Train()
        {
            //Train model
            _model = _trainingPipeline.Fit(_data);
        }
        public SpamPrediction Predict(SpamInput input)
        {
            var predictor=mlContext.Model.CreatePredictionEngine<SpamInput, SpamPrediction>(_model);
            return predictor.Predict(input);
        }
    }
}
