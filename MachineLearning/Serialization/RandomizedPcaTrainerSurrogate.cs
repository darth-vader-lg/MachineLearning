using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class RandomizedPcaTrainerSurrogate
   {
      internal class OptionsSurrogate : UnsupervisedTrainerInputBaseWithWeightSurrogate, ISerializationSurrogate<RandomizedPcaTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (RandomizedPcaTrainer.Options)obj;
            info.AddValue(nameof(data.Rank), data.Rank);
            info.AddValue(nameof(data.Oversampling), data.Oversampling);
            info.AddValue(nameof(data.EnsureZeroMean), data.EnsureZeroMean);
            info.AddValue(nameof(data.Seed), data.Seed);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new RandomizedPcaTrainer.Options();
            SetObjectData(data, info);
            info.Set(nameof(data.Rank), out data.Rank);
            info.Set(nameof(data.Oversampling), out data.Oversampling);
            info.Set(nameof(data.EnsureZeroMean), out data.EnsureZeroMean);
            info.Set(nameof(data.Seed), out data.Seed);
            return data;
         }
      }
   }
}
