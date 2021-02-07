using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Surrogato di serializzazione
   /// </summary>
   internal class LinearSvmTrainerSurrogate
   {
      internal class OptionsSurrogate : OnlineLinearOptionsSurrogate, ISerializationSurrogate<LinearSvmTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (LinearSvmTrainer.Options)obj;
            info.AddValue(nameof(data.Lambda), data.Lambda);
            info.AddValue(nameof(data.BatchSize), data.BatchSize);
            info.AddValue(nameof(data.PerformProjection), data.PerformProjection);
            info.AddValue(nameof(data.NoBias), data.NoBias);
            info.AddValue(nameof(data.ExampleWeightColumnName), data.ExampleWeightColumnName);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new LinearSvmTrainer.Options();
            SetObjectData(data, info);
            info.Set(nameof(data.Lambda), out data.Lambda);
            info.Set(nameof(data.BatchSize), out data.BatchSize);
            info.Set(nameof(data.PerformProjection), out data.PerformProjection);
            info.Set(nameof(data.NoBias), out data.NoBias);
            info.Set(nameof(data.ExampleWeightColumnName), out data.ExampleWeightColumnName);
            return data;
         }
      }
   }
}
