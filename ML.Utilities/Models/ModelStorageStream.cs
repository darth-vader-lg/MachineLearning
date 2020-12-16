using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
      /// <param name="mlContext">Contesto di machine learning</param>
      /// <param name="schema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      public ITransformer LoadModel(MLContext mlContext, out DataViewSchema schema)
      {
         if (ReadStreamGetter == default) {
            schema = default;
            return default;
         }
         return mlContext.Model.Load(ReadStreamGetter(), out schema);
      }
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="model">Modello da salvare</param>
      /// <param name="mlContext">Contesto di machine learning</param>
      /// <param name="schema">Schema di input del modello</param>
      public void SaveModel(MLContext mlContext, ITransformer model, DataViewSchema schema)
      {
         if (WriteStreamGetter == default)
            return;
         mlContext.Model.Save(model, schema, WriteStreamGetter());
      }
      #endregion
   }
}
