using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class SdcaMaximumEntropyMulticlassTrainerSurrogate : SdcaMulticlassTrainerBaseSurrogate<MaximumEntropyModelParameters>
   {
      internal class OptionsSurrogate : MulticlassOptionsSurrogate, ISerializationSurrogate<SdcaMaximumEntropyMulticlassTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new SdcaMaximumEntropyMulticlassTrainer.Options();
            SetObjectData(data, info);
            return data;
         }
      }
   }
}
