using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Surrogato di serializzazione
   /// </summary>
   internal class FieldAwareFactorizationMachineTrainerSurrogate
   {
      internal class OptionsSurrogate : TrainerInputBaseWithWeightSurrogate, ISerializationSurrogate<FieldAwareFactorizationMachineTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (FieldAwareFactorizationMachineTrainer.Options)obj;
            info.AddValue(nameof(data.LearningRate), data.LearningRate);
            info.AddValue(nameof(data.NumberOfIterations), data.NumberOfIterations);
            info.AddValue(nameof(data.LatentDimension), data.LatentDimension);
            info.AddValue(nameof(data.LambdaLinear), data.LambdaLinear);
            info.AddValue(nameof(data.LambdaLatent), data.LambdaLatent);
            info.AddValue(nameof(data.NormalizeFeatures), data.NormalizeFeatures);
            info.AddValue(nameof(data.ExtraFeatureColumns), data.ExtraFeatureColumns);
            info.AddValue(nameof(data.Shuffle), data.Shuffle);
            info.AddValue(nameof(data.Verbose), data.Verbose);
            info.AddValue(nameof(data.Radius), data.Radius);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new FieldAwareFactorizationMachineTrainer.Options();
            SetObjectData(data, info);
            info.Set(nameof(data.LearningRate), out data.LearningRate);
            info.Set(nameof(data.NumberOfIterations), out data.NumberOfIterations);
            info.Set(nameof(data.LatentDimension), out data.LatentDimension);
            info.Set(nameof(data.LambdaLinear), out data.LambdaLinear);
            info.Set(nameof(data.LambdaLatent), out data.LambdaLatent);
            info.Set(nameof(data.NormalizeFeatures), out data.NormalizeFeatures);
            info.Set(nameof(data.ExtraFeatureColumns), out data.ExtraFeatureColumns);
            info.Set(nameof(data.Shuffle), out data.Shuffle);
            info.Set(nameof(data.Verbose), out data.Verbose);
            info.Set(nameof(data.Radius), out data.Radius);
            return data;
         }
      }
   }
}
