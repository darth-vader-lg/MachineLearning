using Microsoft.ML;
using System;
using System.IO;

namespace ML.Utilities.Models
{
   /// <summary>
   /// Gestore dello storage in memoria dei modelli
   /// </summary>
   [Serializable]
   public sealed class ModelStorageMemory : IModelStorage
   {
      #region Properties
      /// <summary>
      /// Bytes del modello
      /// </summary>
      public byte[] Bytes { get; private set; }
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
      /// <param name="mlContext">Contesto di machine learning</param>
      /// <param name="schema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      public ITransformer LoadModel(MLContext mlContext, out DataViewSchema schema)
      {
         using var memoryStream = new MemoryStream(Bytes);
         return mlContext.Model.Load(memoryStream, out schema);
      }
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="model">Modello da salvare</param>
      /// <param name="mlContext">Contesto di machine learning</param>
      /// <param name="schema">Schema di input del modello</param>
      public void SaveModel(MLContext mlContext, ITransformer model, DataViewSchema schema)
      {
         using var memoryStream = new MemoryStream();
         mlContext.Model.Save(model, schema, memoryStream);
         Bytes = memoryStream.ToArray();
      }
      #endregion
   }
}
