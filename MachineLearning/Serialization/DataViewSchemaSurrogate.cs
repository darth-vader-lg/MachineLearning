using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Runtime.Serialization;

namespace MachineLearning.Serialization
{
   internal class DataViewSchemaSurrogate : ISerializationSurrogate<DataViewSchema>
   {
      public void GetObjectData(object obj, SerializationInfo info, StreamingContext context)
      {
         using var stream = new MemoryStream();
         var ml = (context.Context as IMachineLearningContextProvider)?.ML.NET ?? MachineLearningContext.Default.NET;
         ml.Model.Save(new TransformerChain<ITransformer>(), (DataViewSchema)obj, stream);
         info.AddValue("Schema", stream.ToArray(), typeof(byte[]));
      }
      public object SetObjectData(object obj, SerializationInfo info, StreamingContext context, ISurrogateSelector selector)
      {
         var bytes = (byte[])info.GetValue("Schema", typeof(byte[]));
         using var stream = new MemoryStream(bytes);
         var ml = (context.Context as IMachineLearningContextProvider)?.ML.NET ?? MachineLearningContext.Default.NET;
         ml.Model.Load(stream, out var schema);
         return schema;
      }
   }
}
