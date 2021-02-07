using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class SdcaBinaryTrainerBaseSurrogate<TModelParameters> :
      SdcaTrainerBaseSurrogate<
         SdcaBinaryTrainerBase<TModelParameters>.BinaryOptionsBase,
         BinaryPredictionTransformer<TModelParameters>,
         TModelParameters>
      where TModelParameters : class
   {
      internal abstract class BinaryOptionsBaseSurrogate : OptionsBaseSurrogate
      {
         protected static new void GetObjectData(object obj, SerializationInfo info)
         {
            OptionsBaseSurrogate.GetObjectData(obj, info);
            var data = (SdcaBinaryTrainerBase<TModelParameters>.BinaryOptionsBase)obj;
            info.AddValue(nameof(data.PositiveInstanceWeight), data.PositiveInstanceWeight);
         }
         public static new object SetObjectData(object obj, SerializationInfo info)
         {
            var data = (SdcaBinaryTrainerBase<TModelParameters>.BinaryOptionsBase)OptionsBaseSurrogate.SetObjectData(obj, info);
            info.Set(nameof(data.PositiveInstanceWeight), out data.PositiveInstanceWeight);
            return data;
         }
      }
   }
}
