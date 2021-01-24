using MachineLearning.Data;
using Microsoft.ML;
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
      /// <typeparam name="T">Il tipo di contesto</typeparam>
      /// <param name="context">Contesto</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      public ITransformer LoadModel(IMachineLearningContextProvider context, out DataViewSchema inputSchema)
      {
         if (Bytes == null) {
            inputSchema = null;
            return null;
         }
         MachineLearningContext.CheckMLNET(context, nameof(context));
         using var memoryStream = new MemoryStream(Bytes);
         return context.ML.NET.Model.Load(memoryStream, out inputSchema);
      }
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <typeparam name="T">Il tipo di contesto</typeparam>
      /// <param name="context">Contesto</param>
      /// <param name="model">Modello da salvare</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      public void SaveModel(IMachineLearningContextProvider context, ITransformer model, DataViewSchema inputSchema)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         lock (this) {
            var timestamp = DateTime.UtcNow;
            using var memoryStream = new MemoryStream();
            context.ML.NET.Model.Save(model, inputSchema, memoryStream);
            Bytes = memoryStream.ToArray();
            DataTimestamp = timestamp;
         }
      }
      #endregion
   }
}
