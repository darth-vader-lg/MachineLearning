using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Vision;
using System;
using System.IO;
using System.Runtime.Serialization;
using MLV = Microsoft.ML.Vision;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaNonCalibratedMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed partial class ImageClassificationTrainer :
      TrainerBase<ImageClassificationModelParameters, MLV.ImageClassificationTrainer, MLV.ImageClassificationTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml"></param>
      internal ImageClassificationTrainer(MachineLearningContext ml, MLV.ImageClassificationTrainer.Options options = default) : base(ml, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override MLV.ImageClassificationTrainer CreateTrainer(MachineLearningContext ml) => ml.NET.MulticlassClassification.Trainers.ImageClassification(Options);
      #endregion
   }

   /// <summary>
   /// Surrogato delle opzioni per la serializzazione
   /// </summary>
   public partial class ImageClassificationTrainer
   {
      internal class OptionsSurrogate : ISerializationSurrogate
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            var data = (MLV.ImageClassificationTrainer.Options)obj;
            info.AddValue(nameof(data.Arch), (int)data.Arch);
            info.AddValue(nameof(data.BatchSize), data.BatchSize);
            info.AddValue($"{nameof(data.EarlyStoppingCriteria)}.Exists", data.EarlyStoppingCriteria != null);
            if (data.EarlyStoppingCriteria != null) {
               info.AddValue(nameof(data.EarlyStoppingCriteria.CheckIncreasing), data.EarlyStoppingCriteria.CheckIncreasing);
               info.AddValue(nameof(data.EarlyStoppingCriteria.MinDelta), data.EarlyStoppingCriteria.MinDelta);
               info.AddValue(nameof(data.EarlyStoppingCriteria.Patience), data.EarlyStoppingCriteria.Patience);
            }
            info.AddValue(nameof(data.Epoch), data.Epoch);
            info.AddValue(nameof(data.FeatureColumnName), data.FeatureColumnName);
            info.AddValue(nameof(data.FinalModelPrefix), data.FinalModelPrefix);
            info.AddValue(nameof(data.LabelColumnName), data.LabelColumnName);
            info.AddValue(nameof(data.LearningRate), data.LearningRate);
            if (data.LearningRateScheduler != null) {
               if (typeof(LsrDecay).IsAssignableFrom(data.LearningRateScheduler.GetType())) {
                  var scheduler = (LsrDecay)data.LearningRateScheduler;
                  info.AddValue($"{nameof(data.LearningRateScheduler)}.Type", data.LearningRateScheduler.GetType().FullName);
                  info.AddValue($"{nameof(data.LearningRateScheduler)}.{nameof(LsrDecay.BaseLearningRate)}", scheduler.BaseLearningRate);
               }
               else if (typeof(ExponentialLRDecay).IsAssignableFrom(data.LearningRateScheduler.GetType())) {
                  var scheduler = (ExponentialLRDecay)data.LearningRateScheduler;
                  info.AddValue($"{nameof(data.LearningRateScheduler)}.Type", data.LearningRateScheduler.GetType().FullName);
                  info.AddValue($"{nameof(data.LearningRateScheduler)}.{nameof(ExponentialLRDecay.DecayRate)}", scheduler.DecayRate);
                  info.AddValue($"{nameof(data.LearningRateScheduler)}.{nameof(ExponentialLRDecay.DecaySteps)}", scheduler.DecaySteps);
                  info.AddValue($"{nameof(data.LearningRateScheduler)}.{nameof(ExponentialLRDecay.GlobalStep)}", scheduler.GlobalStep);
                  info.AddValue($"{nameof(data.LearningRateScheduler)}.{nameof(ExponentialLRDecay.LearningRate)}", scheduler.LearningRate);
                  info.AddValue($"{nameof(data.LearningRateScheduler)}.{nameof(ExponentialLRDecay.NumEpochsPerDecay)}", scheduler.NumEpochsPerDecay);
                  info.AddValue($"{nameof(data.LearningRateScheduler)}.{nameof(ExponentialLRDecay.Staircase)}", scheduler.Staircase);
               }
               else if (typeof(PolynomialLRDecay).IsAssignableFrom(data.LearningRateScheduler.GetType())) {
                  var scheduler = (PolynomialLRDecay)data.LearningRateScheduler;
                  info.AddValue($"{nameof(data.LearningRateScheduler)}.Type", data.LearningRateScheduler.GetType().FullName);
                  info.AddValue($"{nameof(data.LearningRateScheduler)}.{nameof(PolynomialLRDecay.Cycle)}", scheduler.Cycle);
                  info.AddValue($"{nameof(data.LearningRateScheduler)}.{nameof(PolynomialLRDecay.EndLearningRate)}", scheduler.EndLearningRate);
                  info.AddValue($"{nameof(data.LearningRateScheduler)}.{nameof(PolynomialLRDecay.LearningRate)}", scheduler.LearningRate);
                  info.AddValue($"{nameof(data.LearningRateScheduler)}.{nameof(PolynomialLRDecay.NumEpochsPerDecay)}", scheduler.NumEpochsPerDecay);
                  info.AddValue($"{nameof(data.LearningRateScheduler)}.{nameof(PolynomialLRDecay.Power)}", scheduler.Power);
               }
               else
                  info.AddValue($"{nameof(data.LearningRateScheduler)}.Type", null);
            }
            else
               info.AddValue($"{nameof(data.LearningRateScheduler)}.Type", null);
            //info.AddValue(nameof(data.MetricsCallback), data.MetricsCallback);
            info.AddValue(nameof(data.PredictedLabelColumnName), data.PredictedLabelColumnName);
            info.AddValue(nameof(data.ReuseTrainSetBottleneckCachedValues), data.ReuseTrainSetBottleneckCachedValues);
            info.AddValue(nameof(data.ReuseValidationSetBottleneckCachedValues), data.ReuseValidationSetBottleneckCachedValues);
            info.AddValue(nameof(data.ScoreColumnName), data.ScoreColumnName);
            info.AddValue(nameof(data.TestOnTrainSet), data.TestOnTrainSet);
            info.AddValue(nameof(data.TrainSetBottleneckCachedValuesFileName), data.TrainSetBottleneckCachedValuesFileName);
            if (data.ValidationSet != null) {
               var ml = new MLContext();
               using var ms = new MemoryStream();
               ml.Data.SaveAsBinary(data.ValidationSet, ms, true);
               info.AddValue(nameof(data.ValidationSet), ms.ToArray());
            }
            else
               info.AddValue(nameof(data.ValidationSet), null);
            info.AddValue(nameof(data.ValidationSetBottleneckCachedValuesFileName), data.ValidationSetBottleneckCachedValuesFileName);
            info.AddValue(nameof(data.ValidationSetFraction), data.ValidationSetFraction);
            info.AddValue(nameof(data.WorkspacePath), data.WorkspacePath);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new MLV.ImageClassificationTrainer.Options();
            data.Arch = (MLV.ImageClassificationTrainer.Architecture)info.GetValue(nameof(data.Arch), typeof(int));
            data.BatchSize = (int)info.GetValue(nameof(data.BatchSize), typeof(int));
            if (info.GetBoolean($"{nameof(data.EarlyStoppingCriteria)}.Exists")) {
               data.EarlyStoppingCriteria = new MLV.ImageClassificationTrainer.EarlyStopping(
                  (float)info.GetValue(nameof(data.EarlyStoppingCriteria.MinDelta), typeof(float)),
                  (int)info.GetValue(nameof(data.EarlyStoppingCriteria.Patience), typeof(int)),
                  info.GetBoolean(nameof(data.EarlyStoppingCriteria.CheckIncreasing)) ? Microsoft.ML.Vision.ImageClassificationTrainer.EarlyStoppingMetric.Accuracy : Microsoft.ML.Vision.ImageClassificationTrainer.EarlyStoppingMetric.Loss,
                  (bool)info.GetValue(nameof(data.EarlyStoppingCriteria.CheckIncreasing), typeof(bool)));

            }
            data.Epoch = (int)info.GetValue(nameof(data.Epoch), typeof(int));
            data.FeatureColumnName = (string)info.GetValue(nameof(data.FeatureColumnName), typeof(string));
            data.FinalModelPrefix = (string)info.GetValue(nameof(data.FinalModelPrefix), typeof(string));
            data.LabelColumnName = (string)info.GetValue(nameof(data.LabelColumnName), typeof(string));
            data.LearningRate = (float)info.GetValue(nameof(data.LearningRate), typeof(float));
            var learningRateSchedulerTypeName = (string)info.GetValue($"{nameof(data.LearningRateScheduler)}.Type", typeof(string));
            var learningRateSchedulerType = Type.GetType(learningRateSchedulerTypeName, null, (assembly, typeName, casing) => typeof(LearningRateScheduler).Assembly.GetType(typeName, false, casing));
            if (learningRateSchedulerType != null) {
               if (typeof(LsrDecay).IsAssignableFrom(learningRateSchedulerType))
                  data.LearningRateScheduler = new LsrDecay((float)info.GetValue($"{nameof(data.LearningRateScheduler)}.{nameof(LsrDecay.BaseLearningRate)}", typeof(float)));
               else if (typeof(ExponentialLRDecay).IsAssignableFrom(learningRateSchedulerType)) {
                  data.LearningRateScheduler = new ExponentialLRDecay(
                     (float)info.GetValue($"{nameof(data.LearningRateScheduler)}.{nameof(ExponentialLRDecay.LearningRate)}", typeof(float)),
                     (float)info.GetValue($"{nameof(data.LearningRateScheduler)}.{nameof(ExponentialLRDecay.NumEpochsPerDecay)}", typeof(float)),
                     (float)info.GetValue($"{nameof(data.LearningRateScheduler)}.{nameof(ExponentialLRDecay.DecayRate)}", typeof(float)),
                     (bool)info.GetValue($"{nameof(data.LearningRateScheduler)}.{nameof(ExponentialLRDecay.Staircase)}", typeof(bool)))
                  {
                     GlobalStep = (int)info.GetValue($"{nameof(data.LearningRateScheduler)}.{nameof(ExponentialLRDecay.GlobalStep)}", typeof(int)),
                     DecaySteps = (int)info.GetValue($"{nameof(data.LearningRateScheduler)}.{nameof(ExponentialLRDecay.DecaySteps)}", typeof(int)),
                  };
               }
               else if (typeof(PolynomialLRDecay).IsAssignableFrom(learningRateSchedulerType)) {
                  data.LearningRateScheduler = new PolynomialLRDecay(
                     (float)info.GetValue($"{nameof(data.LearningRateScheduler)}.{nameof(PolynomialLRDecay.LearningRate)}", typeof(float)),
                     (float)info.GetValue($"{nameof(data.LearningRateScheduler)}.{nameof(PolynomialLRDecay.NumEpochsPerDecay)}", typeof(float)),
                     (float)info.GetValue($"{nameof(data.LearningRateScheduler)}.{nameof(PolynomialLRDecay.EndLearningRate)}", typeof(float)),
                     (float)info.GetValue($"{nameof(data.LearningRateScheduler)}.{nameof(PolynomialLRDecay.Power)}", typeof(float)),
                     (bool)info.GetValue($"{nameof(data.LearningRateScheduler)}.{nameof(PolynomialLRDecay.Cycle)}", typeof(bool)));
               }
               else
                  data.LearningRateScheduler = null;
            }
            else
               data.LearningRateScheduler = null;
            //data.MetricsCallback = (Action<ImageClassificationTrainer.ImageClassificationMetrics>)info.GetValue(nameof(data.MetricsCallback), typeof(Action<ImageClassificationTrainer.ImageClassificationMetrics>));
            data.PredictedLabelColumnName = (string)info.GetValue(nameof(data.PredictedLabelColumnName), typeof(string));
            data.ReuseTrainSetBottleneckCachedValues = (bool)info.GetValue(nameof(data.ReuseTrainSetBottleneckCachedValues), typeof(bool));
            data.ReuseValidationSetBottleneckCachedValues = (bool)info.GetValue(nameof(data.ReuseValidationSetBottleneckCachedValues), typeof(bool));
            data.ScoreColumnName = (string)info.GetValue(nameof(data.ScoreColumnName), typeof(string));
            data.TestOnTrainSet = (bool)info.GetValue(nameof(data.TestOnTrainSet), typeof(bool));
            data.TrainSetBottleneckCachedValuesFileName = (string)info.GetValue(nameof(data.TrainSetBottleneckCachedValuesFileName), typeof(string));
            var validationSet = (byte[])info.GetValue(nameof(data.ValidationSet), typeof(byte[]));
            if (validationSet != null) {
               var ml = new MachineLearningContext();
               var storage = new DataStorageBinaryMemory() { BinaryData = (byte[])info.GetValue(nameof(data.ValidationSet), typeof(byte[])) };
               data.ValidationSet = storage.LoadData(this);
            }
            else
               data.ValidationSet = null;
            data.ValidationSetBottleneckCachedValuesFileName = (string)info.GetValue(nameof(data.ValidationSetBottleneckCachedValuesFileName), typeof(string));
            data.ValidationSetFraction = (float?)info.GetValue(nameof(data.ValidationSetFraction), typeof(float?));
            data.WorkspacePath = (string)info.GetValue(nameof(data.WorkspacePath), typeof(string));
            return data;
         }
      }
   }
}
