using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class LbfgsMaximumEntropyMulticlassTrainerSurrogate : LbfgsTrainerBaseSurrogate<LbfgsMaximumEntropyMulticlassTrainer.Options, MulticlassPredictionTransformer<MaximumEntropyModelParameters>, MaximumEntropyModelParameters>
   {
      internal class OptionsSurrogate : OptionsBaseSurrogate, ISerializationSurrogate<LbfgsMaximumEntropyMulticlassTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (LbfgsMaximumEntropyMulticlassTrainer.Options)obj;
            info.AddValue(nameof(data.ShowTrainingStatistics), data.ShowTrainingStatistics);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new LbfgsMaximumEntropyMulticlassTrainer.Options();
            SetObjectData(data, info);
            info.Set(nameof(data.ShowTrainingStatistics), out data.ShowTrainingStatistics);
            return data;
         }
      }
   }
}
