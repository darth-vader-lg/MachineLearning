using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class SdcaMulticlassTrainerBaseSurrogate<TModel> :
      SdcaTrainerBaseSurrogate<SdcaMulticlassTrainerBase<TModel>.MulticlassOptions, MulticlassPredictionTransformer<TModel>, TModel>
      where TModel : class
   {
      internal class MulticlassOptionsSurrogate
      {
         private static OptionsBaseSurrogate Base => new OptionsBaseSurrogate();
         public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            Base.GetObjectData(obj, info, context);
         }
         public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            return Base.SetObjectData(obj, info, context, selector);
         }
      }
   }
}
