using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class SdcaMulticlassTrainerBaseSurrogate<TModel> :
      SdcaTrainerBaseSurrogate<SdcaMulticlassTrainerBase<TModel>.MulticlassOptions, MulticlassPredictionTransformer<TModel>, TModel>
      where TModel : class
   {
      internal abstract class MulticlassOptionsSurrogate : OptionsBaseSurrogate
      {
         protected static new void GetObjectData(object obj, SerializationInfo info) => OptionsBaseSurrogate.GetObjectData(obj, info);
         protected static new  object SetObjectData(object obj, SerializationInfo info) => OptionsBaseSurrogate.SetObjectData(obj, info);
      }
   }
}
