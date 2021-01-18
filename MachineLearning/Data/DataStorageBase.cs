using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.IO;
using System.Linq;

namespace MachineLearning.Data
{
   /// <summary>
   /// Classe base per lo storage di dati
   /// </summary>
   [Serializable]
   public abstract class DataStorageBase : IDisposable, IMultiStreamSource
   {
      #region Properties
      /// <summary>
      /// Il numero di items
      /// </summary>
      int IMultiStreamSource.Count => 1;
      /// <summary>
      /// Indicatore di oggetto disposed
      /// </summary>
      public bool IsDisposed { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Dispose da programma
      /// </summary>
      public void Dispose()
      {
         Dispose(disposing: true);
         GC.SuppressFinalize(this);
      }
      /// <summary>
      /// Funzione di dispose
      /// </summary>
      /// <param name="disposing">Indicatore di dispose da programma</param>
      protected virtual void Dispose(bool disposing) => IsDisposed = true;
      /// <summary>
      /// Restituisce una stringa rappresentante il "path" dello stream. Puo' essere null.
      /// </summary>
      /// <returns>Il path o null</returns>
      protected virtual string GetFilePath() => null;
      /// <summary>
      /// Restituisce uno stream leggibile.
      /// </summary>
      /// <returns>Lo stream di lettura</returns>
      protected virtual Stream GetReadStream() => null;
      /// <summary>
      /// Restituisce una stringa rappresentante il "path" dello stream indicato da index. Potrebbe essere null.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Il path</returns>
      string IMultiStreamSource.GetPathOrNull(int index) => GetFilePath();
      /// <summary>
      /// Apre l'item indicato e ne restituisce uno stream leggibile.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Lo stream di lettura</returns>
      Stream IMultiStreamSource.Open(int index) => GetReadStream();
      /// <summary>
      /// Apre l'item indicato e ne restituisce uno stream di stringhe leggibile.
      /// </summary>
      /// <param name="index">L'indice dell'item</param>
      /// <returns>Lo stream di lettura</returns>
      TextReader IMultiStreamSource.OpenTextReader(int index) => new StreamReader(GetReadStream());
      /// <summary>
      /// Carica i dati in formato binario
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <returns>L'accesso ai dati</returns>
      protected IDataAccess LoadBinaryData(IMachineLearningContextProvider context)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         return new DataAccess(context, context.ML.NET.Data.LoadFromBinary(this));
      }
      /// <summary>
      /// Carica i dati in formato testo
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <returns>L'accesso ai dati</returns>
      protected IDataAccess LoadTextData(IMachineLearningContextProvider context)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         context.ML.NET.CheckParam(context is ITextLoaderOptionsProvider, nameof(context), $"The context is not a {typeof(ITextLoaderOptionsProvider).Name}");
         var opt = (context as ITextLoaderOptionsProvider)?.TextLoaderOptions ?? new TextLoader.Options();
         return new DataAccess(context, context.ML.NET.Data.CreateTextLoader(opt).Load(this));
      }
      /// <summary>
      /// Salva i dati in formato binario
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="stream">Stream per la scrittura. Lo stream viene chiuso automaticamente al termine della scrittura</param>
      /// <param name="keepHidden">Se mantenere le colonne nascoste nel set di dati</param>
      protected void SaveBinaryData(IMachineLearningContextProvider context, IDataView data, Stream stream, bool keepHidden = false)
      {
         lock (this) {
            try {
               if (stream == null)
                  throw new ArgumentException($"{nameof(stream)} cannot be null");
               if (!stream.CanWrite)
                  throw new ArgumentException($"{nameof(stream)} must be writable");
               // Contesto
               (context?.ML?.NET ?? new MLContext()).Data.SaveAsBinary(data, stream, keepHidden);
            }
            finally {
               try {
                  if (stream != null)
                     stream.Close();
               }
               catch (Exception) { }
            }
         }
      }
      /// <summary>
      /// Salva i dati in formato testo
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="stream">Stream per la scrittura. Lo stream viene chiuso automaticamente al termine della scrittura a meno che non specificato</param>
      /// <param name="schema">Specifica se salvare il commento con lo schema all'inizio del file</param>
      /// <param name="keepHidden">Specifica se mantenere le colonne nascoste nel set di dati</param>
      /// <param name="forceDense">Specifica se salvare le colonne in formato denso anche se sono vettori sparsi</param>
      /// <param name="closeStream">Determina se chiudere lo stream al termine della scrittura</param>
      protected void SaveTextData(IMachineLearningContextProvider context, IDataView data, Stream stream, bool schema = false, bool keepHidden = false, bool forceDense = false, bool closeStream = true)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         lock (this) {
            try {
               // Verifica del writer
               if (stream == null)
                  throw new InvalidOperationException("Cannot save the data");
               if (!stream.CanWrite)
                  throw new ArgumentException($"{nameof(stream)} must be writable");
               // Contesto e opzioni
               var ml = context?.ML?.NET ?? new MLContext();
               var opt = (context as ITextLoaderOptionsProvider)?.TextLoaderOptions ?? new TextLoader.Options();
               // Separatore di colonne
               var separator = opt.Separators?.FirstOrDefault() ?? '\t';
               separator = separator != default ? separator : '\t';
               // Salva come testo i dati
               ml.Data.SaveAsText(
                  data: data,
                  stream: stream,
                  separatorChar: separator,
                  headerRow: opt.HasHeader,
                  schema: schema,
                  keepHidden: keepHidden,
                  forceDense: forceDense);
            }
            finally {
               try {
                  if (stream != default && closeStream)
                     stream.Close();
               }
               catch (Exception) { }
            }
         }
      }
      #endregion
   }
}
