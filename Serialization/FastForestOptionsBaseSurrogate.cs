using Microsoft.ML.Trainers.FastTree;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal abstract class FastForestOptionsBaseSurrogate : TreeOptionsSurrogate
   {
      protected static new void GetObjectData(object obj, SerializationInfo info)
      {
         TreeOptionsSurrogate.GetObjectData(obj, info);
         var data = (FastForestOptionsBase)obj;
         info.AddValue(nameof(data.NumberOfQuantileSamples), data.NumberOfQuantileSamples);
      }
      protected static new object SetObjectData(object obj, SerializationInfo info)
      {
         var data = (FastForestOptionsBase)TreeOptionsSurrogate.SetObjectData(obj, info);
         info.Set(nameof(data.NumberOfQuantileSamples), out data.NumberOfQuantileSamples);
         return data;
      }
   }
}
