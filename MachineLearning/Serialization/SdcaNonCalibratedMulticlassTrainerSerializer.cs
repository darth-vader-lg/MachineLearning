﻿using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Classe helper per la serializzazione dei dati del trainer
   /// </summary>
   internal class SdcaNonCalibratedMulticlassTrainerSerializer
   {
      internal class Options
      {
         internal class SerializationSurrogate : ISerializationSurrogate
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
               var data = (SdcaNonCalibratedMulticlassTrainer.Options)obj;
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
}
