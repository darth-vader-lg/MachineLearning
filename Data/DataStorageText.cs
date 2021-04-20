using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.IO;
using System.Linq;

namespace MachineLearning.Data
{
   /// <summary>
   /// Classe base per lo storage di dati di tipo file di testo
   /// </summary>
   [Serializable]
   public abstract class DataStorageText : DataStorageBase, IDataStorage
   {
      #region Properties
      /// <summary>
      /// Specifica se salvare il commento con lo schema all'inizio del file
      /// </summary>
      public bool SaveSchema { get; set; }
      /// <summary>
      /// Specifica se mantenere le colonne nascoste nel set di dati
      /// </summary>
      public bool KeepHidden { get; set; }
      /// <summary>
      /// Specifica se salvare le colonne in formato denso anche se sono vettori sparsi
      /// </summary>
      public bool ForceDense { get; set; }
      #endregion
      #region Methods
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="textLoaderOptions">Opzioni di caricamento testuale</param>
      /// <returns>L'accesso ai dati</returns>
      public virtual IDataAccess LoadData(MLContext context, TextLoader.Options textLoaderOptions) =>
         LoadTextData(context, textLoaderOptions);
      /// <summary>
      /// Carica i dati in formato testo
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="textLoaderOptions">Opzioni di caricamento testuale</param>
      /// <returns>L'accesso ai dati</returns>
      protected IDataAccess LoadTextData(MLContext context, TextLoader.Options textLoaderOptions)
      {
         Contracts.CheckValue(context, nameof(context));
         context.CheckValue(textLoaderOptions, nameof(textLoaderOptions));
         return new DataAccess(context, context.Data.CreateTextLoader(textLoaderOptions).Load(this));
      }
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="textLoaderOptions">Opzioni di caricamento testuale</param>
      public abstract void SaveData(MLContext context, IDataAccess data, TextLoader.Options textLoaderOptions);
      /// <summary>
      /// Salva i dati in formato testo
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="options">Opzioni di caricamento testuale</param>
      /// <param name="stream">Stream per la scrittura. Lo stream viene chiuso automaticamente al termine della scrittura a meno che non specificato</param>
      /// <param name="closeStream">Determina se chiudere lo stream al termine della scrittura</param>
      protected void SaveTextData(MLContext context, IDataAccess data, TextLoader.Options options, Stream stream, bool closeStream = true)
      {
         Contracts.CheckValue(context, nameof(context));
         context.CheckValue(options, nameof(options));
         lock (this) {
            try {
               // Verifica del writer
               if (stream == null)
                  throw new InvalidOperationException("Cannot save the data");
               if (!stream.CanWrite)
                  throw new ArgumentException($"{nameof(stream)} must be writable");
               // Separatore di colonne
               var separator = options.Separators?.FirstOrDefault() ?? '\t';
               separator = separator != default ? separator : '\t';
               // Salva come testo i dati
               context.Data.SaveAsText(
                  data: data,
                  stream: stream,
                  separatorChar: separator,
                  headerRow: options.HasHeader,
                  schema: SaveSchema,
                  keepHidden: KeepHidden,
                  forceDense: ForceDense);
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
