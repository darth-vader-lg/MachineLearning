using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class SdcaRegressionTrainerSurrogate : SdcaTrainerBaseSurrogate<SdcaRegressionTrainer.Options, RegressionPredictionTransformer<LinearRegressionModelParameters>, LinearRegressionModelParameters>
   {
      internal class OptionsSurrogate : ISerializationSurrogate<SdcaRegressionTrainer.Options>
      {
         private static OptionsBaseSurrogate Base => new OptionsBaseSurrogate();
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            Base.GetObjectData(obj, info, context);
            var data = (SdcaRegressionTrainer.Options)obj;
            info.AddValue(nameof(data.LossFunction), data.LossFunction);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = new SdcaRegressionTrainer.Options();
            Base.SetObjectData(obj, info, context, selector);
            info.Set(nameof(data.LossFunction), () => data.LossFunction, value => { if (value != null) data.LossFunction = value; });
            return data;
         }
      }
   }
}
