using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class SdcaNonCalibratedMulticlassTrainerSurrogate : SdcaMulticlassTrainerBaseSurrogate<LinearMulticlassModelParameters>
   {
      internal class OptionsSurrogate : MulticlassOptionsSurrogate, ISerializationSurrogate<SdcaNonCalibratedMulticlassTrainer.Options>
      {
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            GetObjectData(obj, info);
            var data = (SdcaNonCalibratedMulticlassTrainer.Options)obj;
            info.AddValue(nameof(data.Loss), data.Loss);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new SdcaNonCalibratedMulticlassTrainer.Options();
            SetObjectData(obj = data, info);
            info.Set(nameof(data.Loss), () => data.Loss, value => { if (value != null) data.Loss = value; });
            return data;
         }
      }
   }
}
