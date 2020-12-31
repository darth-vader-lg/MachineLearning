using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.Runtime.Serialization;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe SdcaNonCalibratedMulticlassTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed partial class SdcaNonCalibratedMulticlass :
      TrainerBase<LinearMulticlassModelParameters, SdcaNonCalibratedMulticlassTrainer, SdcaNonCalibratedMulticlassTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      internal SdcaNonCalibratedMulticlass(MachineLearningContext ml, SdcaNonCalibratedMulticlassTrainer.Options options = default) : base(ml, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override SdcaNonCalibratedMulticlassTrainer CreateTrainer(MachineLearningContext ml) => ml.NET.MulticlassClassification.Trainers.SdcaNonCalibrated(Options);
      #endregion
   }

   /// <summary>
   /// Surrogato delle opzioni per la serializzazione
   /// </summary>
   public partial class SdcaNonCalibratedMulticlass
   {
      internal class OptionsSurrogate : ISerializationSurrogate
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            var data = (SdcaNonCalibratedMulticlassTrainer.Options)obj;
            info.AddValue(nameof(data.BiasLearningRate), data.BiasLearningRate);
            info.AddValue(nameof(data.ConvergenceCheckFrequency), data.ConvergenceCheckFrequency);
            info.AddValue(nameof(data.ConvergenceTolerance), data.ConvergenceTolerance);
            info.AddValue(nameof(data.ExampleWeightColumnName), data.ExampleWeightColumnName);
            info.AddValue(nameof(data.FeatureColumnName), data.FeatureColumnName);
            info.AddValue(nameof(data.L1Regularization), data.L1Regularization);
            info.AddValue(nameof(data.L2Regularization), data.L2Regularization);
            info.AddValue(nameof(data.LabelColumnName), data.LabelColumnName);
            info.AddValue(nameof(data.Loss), data.Loss);
            info.AddValue(nameof(data.MaximumNumberOfIterations), data.MaximumNumberOfIterations);
            info.AddValue(nameof(data.NumberOfThreads), data.NumberOfThreads);
            info.AddValue(nameof(data.Shuffle), data.Shuffle);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new SdcaNonCalibratedMulticlassTrainer.Options();
            data.BiasLearningRate = (float)info.GetValue(nameof(data.BiasLearningRate), typeof(float));
            data.ConvergenceCheckFrequency = (int?)info.GetValue(nameof(data.ConvergenceCheckFrequency), typeof(int?));
            data.ConvergenceTolerance = (float)info.GetValue(nameof(data.ConvergenceTolerance), typeof(float));
            data.ExampleWeightColumnName = (string)info.GetValue(nameof(data.ExampleWeightColumnName), typeof(string));
            data.FeatureColumnName = (string)info.GetValue(nameof(data.FeatureColumnName), typeof(string));
            data.L1Regularization = (float?)info.GetValue(nameof(data.L1Regularization), typeof(float?));
            data.L2Regularization = (float?)info.GetValue(nameof(data.L2Regularization), typeof(float?));
            data.LabelColumnName = (string)info.GetValue(nameof(data.LabelColumnName), typeof(string));
            data.Loss = (ISupportSdcaClassificationLoss)info.GetValue(nameof(data.Loss), typeof(ISupportSdcaClassificationLoss));
            data.MaximumNumberOfIterations = (int?)info.GetValue(nameof(data.MaximumNumberOfIterations), typeof(int?));
            data.NumberOfThreads = (int?)info.GetValue(nameof(data.NumberOfThreads), typeof(int?));
            data.Shuffle = (bool)info.GetValue(nameof(data.Shuffle), typeof(bool));
            return data;
         }
      }
   }
}
