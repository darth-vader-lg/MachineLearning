using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   /// <summary>
   /// Surrogato di serializzazione
   /// </summary>
   internal class LdSvmTrainerSurrogate
   {
      internal class OptionsSurrogate : TrainerInputBaseWithWeightSurrogate, ISerializationSurrogate<LdSvmTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (LdSvmTrainer.Options)obj;
            info.AddValue(nameof(data.TreeDepth), data.TreeDepth);
            info.AddValue(nameof(data.LambdaW), data.LambdaW);
            info.AddValue(nameof(data.LambdaTheta), data.LambdaTheta);
            info.AddValue(nameof(data.LambdaThetaprime), data.LambdaThetaprime);
            info.AddValue(nameof(data.Sigma), data.Sigma);
            info.AddValue(nameof(data.UseBias), data.UseBias);
            info.AddValue(nameof(data.NumberOfIterations), data.NumberOfIterations);
            info.AddValue(nameof(data.Cache), data.Cache);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new LdSvmTrainer.Options();
            SetObjectData(data, info);
            info.Set(nameof(data.TreeDepth), out data.TreeDepth);
            info.Set(nameof(data.LambdaW), out data.LambdaW);
            info.Set(nameof(data.LambdaTheta), out data.LambdaTheta);
            info.Set(nameof(data.LambdaThetaprime), out data.LambdaThetaprime);
            info.Set(nameof(data.Sigma), out data.Sigma);
            info.Set(nameof(data.UseBias), out data.UseBias);
            info.Set(nameof(data.NumberOfIterations), out data.NumberOfIterations);
            info.Set(nameof(data.Cache), out data.Cache);
            return data;
         }
      }
   }
}
