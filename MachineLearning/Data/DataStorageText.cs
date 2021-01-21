using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using System;
using System.IO;
using System.Linq;
using System.Text;

namespace MachineLearning.Data
{
   /// <summary>
   /// Classe base per lo storage di dati di tipo file di testo
   /// </summary>
   [Serializable]
   public abstract class DataStorageText : DataStorageBase, IDataStorage, ITextLoaderOptionsProvider
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
      /// <summary>
      /// Opzioni di caricamento testi
      /// </summary>
      public TextLoader.Options TextLoaderOptions { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="options">Opzioni di caricamento dei testi</param>
      public DataStorageText(TextLoader.Options options)
      {
         Contracts.CheckValue(options, nameof(options));
         TextLoaderOptions = options;
      }
      /// <summary>
      /// Formatta una riga di dati di input da un elenco di dati di input
      /// </summary>
      /// <param name="data">Dati di input</param>
      /// <returns></returns>
      public string FormatDataRow(params string[] data)
      {
         // Linea da passare al modello
         var inputLine = new StringBuilder();
         // Quotatura stringhe
         var quote = TextLoaderOptions.AllowQuoting ? "\"" : "";
         // Separatore di colonne
         var separatorChar = TextLoaderOptions.Separators?.FirstOrDefault() ?? ',';
         // Loop di costruzione della linea di dati
         var separator = "";
         foreach (var item in data) {
            var text = item ?? "";
            var quoting = quote.Length > 0 && text.TrimStart().StartsWith(quote) && text.TrimEnd().EndsWith(quote) ? "" : quote;
            inputLine.Append($"{separator}{quoting}{text}{quoting}");
            separator = new string(separatorChar, 1);
         }
         return inputLine.ToString();
      }
      /// <summary>
      /// Carica i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <returns>L'accesso ai dati</returns>
      public virtual IDataAccess LoadData(IMachineLearningContextProvider context)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         return LoadTextData(context);
      }
      /// <summary>
      /// Carica i dati in formato testo
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <returns>L'accesso ai dati</returns>
      protected IDataAccess LoadTextData(IMachineLearningContextProvider context)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         return new DataAccess(context, context.ML.NET.Data.CreateTextLoader(TextLoaderOptions).Load(this));
      }
      /// <summary>
      /// Salva i dati
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      public abstract void SaveData(IMachineLearningContextProvider context, IDataView data);
      /// <summary>
      /// Salva i dati in formato testo
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="data">L'accesso ai dati</param>
      /// <param name="stream">Stream per la scrittura. Lo stream viene chiuso automaticamente al termine della scrittura a meno che non specificato</param>
      /// <param name="closeStream">Determina se chiudere lo stream al termine della scrittura</param>
      protected void SaveTextData(IMachineLearningContextProvider context, IDataView data, Stream stream, bool closeStream = true)
      {
         MachineLearningContext.CheckMLNET(context, nameof(context));
         lock (this) {
            try {
               // Verifica del writer
               if (stream == null)
                  throw new InvalidOperationException("Cannot save the data");
               if (!stream.CanWrite)
                  throw new ArgumentException($"{nameof(stream)} must be writable");
               // Separatore di colonne
               var separator = TextLoaderOptions.Separators?.FirstOrDefault() ?? '\t';
               separator = separator != default ? separator : '\t';
               // Salva come testo i dati
               context.ML.NET.Data.SaveAsText(
                  data: data,
                  stream: stream,
                  separatorChar: separator,
                  headerRow: TextLoaderOptions.HasHeader,
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
