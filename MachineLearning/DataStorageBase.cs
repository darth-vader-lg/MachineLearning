using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;

namespace MachineLearning
{
   /// <summary>
   /// Classe base per lo storage di dati
   /// </summary>
   [Serializable]
   public abstract class DataStorageBase : IMultiStreamSource
   {
      #region Properties
      /// <summary>
      /// Il numero di items
      /// </summary>
      int IMultiStreamSource.Count => 1;
      #endregion
      #region Methods
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
      protected IDataView LoadBinaryData(object context)
      {
         var ml = (context as IMachineLearningContextProvider)?.ML?.NET ?? new MLContext();
         return ml.Data.LoadFromBinary(this);
      }
      /// <summary>
      /// Carica i dati in formato testo
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <returns>L'accesso ai dati</returns>
      protected IDataView LoadTextData(object context)
      {
         var ml = (context as IMachineLearningContextProvider)?.ML?.NET ?? new MLContext();
         var opt = (context as ITextLoaderOptionsProvider)?.TextLoaderOptions ?? new TextLoader.Options();
         return ml.Data.CreateTextLoader(opt).Load(this);
      }
      /// <summary>
      /// Salva i dati in formato binario
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="stream">Stream per la scrittura. Lo stream viene chiuso automaticamente al termine della scrittura</param>
      /// <param name="keepHidden">Se mantenere le colonne nascoste nel set di dati</param>
      protected void SaveBinaryData(object context, IDataView data, Stream stream, bool keepHidden = false)
      {
         lock (this) {
            try {
               if (stream == null)
                  throw new ArgumentException($"{nameof(stream)} cannot be null");
               if (!stream.CanWrite)
                  throw new ArgumentException($"{nameof(stream)} must be writable");
               // Contesto
               var ml = (context as IMachineLearningContextProvider)?.ML?.NET ?? new MLContext();
               ml.Data.SaveAsBinary(data, stream, keepHidden);
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
      protected void SaveTextData(object context, IDataView data, Stream stream, bool schema = false, bool keepHidden = false, bool forceDense = false, bool closeStream = true)
      {
         lock (this) {
            try {
               // Verifica del writer
               if (stream == null)
                  throw new InvalidOperationException("Cannot save the data");
               if (!stream.CanWrite)
                  throw new ArgumentException($"{nameof(stream)} must be writable");
               // Contesto e opzioni
               var ml = (context as IMachineLearningContextProvider)?.ML?.NET ?? new MLContext();
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
