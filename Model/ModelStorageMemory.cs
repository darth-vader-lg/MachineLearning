using MachineLearning.Data;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using System;
using System.IO;

namespace MachineLearning.Model
{
   /// <summary>
   /// Gestore dello storage in memoria dei modelli
   /// </summary>
   [Serializable]
   public sealed class ModelStorageMemory : IModelStorage, IDataTimestamp
   {
      #region Properties
      /// <summary>
      /// Bytes del modello
      /// </summary>
      public byte[] Bytes { get; private set; }
      /// <summary>
      /// Data e ora dell'oggetto
      /// </summary>
      public DateTime DataTimestamp { get; private set; } = DateTime.UtcNow;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public ModelStorageMemory() { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="memory">Contenuto iniziale</param>
      public ModelStorageMemory(byte[] bytes) => Bytes = bytes;
      /// <summary>
      /// Funzione di caricamento modello
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      public ITransformer LoadModel(MLContext context, out DataViewSchema inputSchema)
      {
         if (Bytes == null) {
            inputSchema = null;
            return null;
         }
         Contracts.CheckValue(context, nameof(context));
         using var memoryStream = new MemoryStream(Bytes);
         return context.Model.Load(memoryStream, out inputSchema);
      }
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="model">Modello da salvare</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      public void SaveModel(MLContext context, ITransformer model, DataViewSchema inputSchema)
      {
         Contracts.CheckValue(context, nameof(context));
         lock (this) {
            var timestamp = DateTime.UtcNow;
            using var memoryStream = new MemoryStream();
            context.Model.Save(model, inputSchema, memoryStream);
            Bytes = memoryStream.ToArray();
            DataTimestamp = timestamp;
         }
      }
      #endregion
   }
}
