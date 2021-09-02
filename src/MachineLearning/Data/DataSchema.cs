using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization;

namespace MachineLearning.Data
{
   /// <summary>
   /// Data schema
   /// </summary>
   [Serializable]
   public class DataSchema :
      IEnumerable,
      IEnumerable<DataViewSchema.Column>,
      IReadOnlyCollection<DataViewSchema.Column>,
      IReadOnlyList<DataViewSchema.Column>,
      ISerializable
   {
      #region Fields
      /// <summary>
      /// Context provider for ML.NET
      /// </summary>
      [NonSerialized]
      private IContextProvider<MLContext> context;
      /// <summary>
      /// Serialization data
      /// </summary>
      [NonSerialized]
      private byte[] serializationData;
      /// <summary>
      /// Data schema in ML-NET format
      /// </summary>
      [NonSerialized]
      private DataViewSchema schema;
      #endregion
      #region Properties
      /// <summary>
      /// Number of columns in the schema.
      /// </summary>
      public int Count => DataViewSchema.Count;
      /// <summary>
      /// Data schema in ML-NET format
      /// </summary>
      public DataViewSchema DataViewSchema
      {
         get
         {
            if (schema == null && context != null && serializationData != null) {
               try {
                  using var stream = new MemoryStream(serializationData);
                  context.Context.Model.Load(stream, out schema);
               }
               finally {
                  context = null;
                  serializationData = null;
               }
            }
            return schema;
         }
      }
      /// <summary>
      /// Get the column by index.
      /// </summary>
      public DataViewSchema.Column this[int index] => DataViewSchema[index];
      /// <summary>
      /// Get the column by name. Throws an exception if such column does not exist. Note
      /// that if multiple columns exist with the same name, the one with the biggest index
      /// is returned. The other columns are considered 'hidden', and only accessible by
      /// their index.
      /// </summary>
      public DataViewSchema.Column this[string name] => DataViewSchema[name];
      #endregion
      /// <summary>
      /// Default contructor
      /// </summary>
      private DataSchema() { }
      /// <summary>
      /// Deserialization constructor
      /// </summary>
      /// <param name="info">Information</param>
      /// <param name="context">Context</param>
      private DataSchema(SerializationInfo info, StreamingContext context)
      {
         // Reset the property value using the GetValue method.
         this.context = (context.Context as IContextProvider<MLContext>) ?? MachineLearningContext.Default;
         serializationData = (byte[])info.GetValue("data", typeof(byte[]));
      }
      /// <summary>
      /// The enumerator
      /// </summary>
      public IEnumerator<DataViewSchema.Column> GetEnumerator() => DataViewSchema.GetEnumerator();
      /// <summary>
      /// The enumerator
      /// </summary>
      IEnumerator IEnumerable.GetEnumerator() => DataViewSchema.GetEnumerator();
      /// <summary>
      /// Serialization data preparation
      /// </summary>
      /// <param name="info">Information</param>
      /// <param name="context">Context</param>
      void ISerializable.GetObjectData(SerializationInfo info, StreamingContext context)
      {
         using var stream = new MemoryStream();
         var ml = (context.Context as IContextProvider<MLContext>) ?? MachineLearningContext.Default;
         ml.Context.Model.Save(new TransformerChain<ITransformer>(), DataViewSchema, stream);
         info.AddValue("data", stream.ToArray(), typeof(byte[]));
      }
      /// <summary>
      /// Convertion from ML.NET DataViewSchema to DataSchema
      /// </summary>
      /// <param name="dataViewSchema">DataViewSchema instance</param>
      public static implicit operator DataSchema(DataViewSchema dataViewSchema) => new() { schema = dataViewSchema };
      /// <summary>
      /// Convertion from DataSchema to ML.NET DataViewSchema
      /// </summary>
      /// <param name="dataSchema">DataSchema instance</param>
      public static implicit operator DataViewSchema(DataSchema dataSchema) => dataSchema.DataViewSchema;
   }
}
