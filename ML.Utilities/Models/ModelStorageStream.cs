using Microsoft.ML;
using System;
using System.IO;

namespace ML.Utilities.Models
{
   /// <summary>
   /// Gestore dello storage su stream dei modelli
   /// </summary>
   [Serializable]
   public sealed class ModelStorageStream : IModelStorage
   {
      #region Properties
      /// <summary>
      /// Funzione di restituzione della stream in lettura
      /// </summary>
      private readonly Func<Stream> ReadStreamGetter;
      /// <summary>
      /// Funzione di restituzione della stream in lettura
      /// </summary>
      private readonly Func<Stream> WriteStreamGetter;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ReadStreamGetter">Funzione di restituzione della stream in lettura</param>
      /// <param name="WriteStreamGetter">Funzione di restituzione della stream in scrittura</param>
      public ModelStorageStream(Func<Stream> ReadStreamGetter = null, Func<Stream> WriteStreamGetter = null)
      {
         this.ReadStreamGetter = ReadStreamGetter;
         this.WriteStreamGetter = WriteStreamGetter;
      }
      /// <summary>
      /// Funzione di caricamento modello
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      public ITransformer LoadModel(MachineLearningContext ml, out DataViewSchema inputSchema)
      {
         if (ReadStreamGetter == default) {
            inputSchema = default;
            return default;
         }
         using var stream = ReadStreamGetter();
         return ml.NET.Model.Load(stream, out inputSchema);
      }
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="model">Modello da salvare</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      public void SaveModel(MachineLearningContext ml, ITransformer model, DataViewSchema inputSchema)
      {
         if (WriteStreamGetter == default)
            return;
         using var stream = WriteStreamGetter();
         ml.NET.Model.Save(model, inputSchema, stream);
      }
      #endregion
   }
}
