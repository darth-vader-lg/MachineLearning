using Microsoft.ML.Trainers.LightGbm;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class BoosterParameterBaseSurrogate
   {
      internal class OptionsBaseSurrogate : ISerializationSurrogate<BoosterParameterBase.OptionsBase>
      {
         public virtual void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
         {
            var data = (BoosterParameterBase.OptionsBase)obj;
            info.AddValue(nameof(data.FeatureFraction), data.FeatureFraction);
            info.AddValue(nameof(data.L1Regularization), data.L1Regularization);
            info.AddValue(nameof(data.L2Regularization), data.L2Regularization);
            info.AddValue(nameof(data.MaximumTreeDepth), data.MaximumTreeDepth);
            info.AddValue(nameof(data.MinimumChildWeight), data.MinimumChildWeight);
            info.AddValue(nameof(data.MinimumSplitGain), data.MinimumSplitGain);
            info.AddValue(nameof(data.SubsampleFraction), data.SubsampleFraction);
            info.AddValue(nameof(data.SubsampleFrequency), data.SubsampleFrequency);
         }
         public virtual object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
         {
            var data = (BoosterParameterBase.OptionsBase)obj;
            void Set<T>(string name, ref T value) => value = (T)info.GetValue(name, typeof(T));
            Set(nameof(data.FeatureFraction), ref data.FeatureFraction);
            Set(nameof(data.L1Regularization), ref data.L1Regularization);
            Set(nameof(data.L2Regularization), ref data.L2Regularization);
            Set(nameof(data.MaximumTreeDepth), ref data.MaximumTreeDepth);
            Set(nameof(data.MinimumChildWeight), ref data.MinimumChildWeight);
            Set(nameof(data.MinimumSplitGain), ref data.MinimumSplitGain);
            Set(nameof(data.SubsampleFraction), ref data.SubsampleFraction);
            Set(nameof(data.SubsampleFrequency), ref data.SubsampleFrequency);
            return data;
         }
      }
   }
}
