using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class SdcaNonCalibratedMulticlassTrainerSurrogate : SdcaMulticlassTrainerBaseSurrogate<LinearMulticlassModelParameters>
   {
      internal class OptionsSurrogate : ISerializationSurrogate<SdcaNonCalibratedMulticlassTrainer.Options>
      {
         private static MulticlassOptionsSurrogate Base => new MulticlassOptionsSurrogate();
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            Base.GetObjectData(obj, info, context);
            var data = (SdcaNonCalibratedMulticlassTrainer.Options)obj;
            info.AddValue(nameof(data.Loss), data.Loss);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new SdcaNonCalibratedMulticlassTrainer.Options();
            Base.SetObjectData(obj, info, context, selector);
            info.Set(nameof(data.Loss), () => data.Loss, value => { if (value != null) data.Loss = value; });
            return data;
         }
      }
   }
}
