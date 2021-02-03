using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class OlsTrainerSurrogate
   {
      internal class OptionsSurrogate : TrainerInputBaseWithWeightSurrogate, ISerializationSurrogate<OlsTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (OlsTrainer.Options)obj;
            info.AddValue(nameof(data.L2Regularization), data.L2Regularization);
            info.AddValue(nameof(data.CalculateStatistics), data.CalculateStatistics);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new OlsTrainer.Options();
            SetObjectData(obj = data, info);
            info.Set(nameof(data.L2Regularization), out data.L2Regularization);
            info.Set(nameof(data.CalculateStatistics), out data.CalculateStatistics);
            return data;
         }
      }
   }
}
