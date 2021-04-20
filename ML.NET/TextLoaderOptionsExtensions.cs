using MachineLearning;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Data
{
   /// <summary>
   /// Estensioni alla classe TextLoader.Options
   /// </summary>
   public static class TextLoaderOptionsExtensions
   {
      #region Methods
      /// <summary>
      /// Formatta una riga di dati di input da un elenco di dati di input
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <param name="data">Dati di input</param>
      /// <returns>La stringa unica formattata</returns>
      public static string FormatDataRow(this TextLoader.Options options, params string[] data)
      {
         // Linea da passare al modello
         var inputLine = new StringBuilder();
         // Quotatura stringhe
         var quote = options.AllowQuoting ? "\"" : "";
         // Separatore di colonne
         var separatorChar = options.Separators?.FirstOrDefault() ?? ',';
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
      /// Splitta i dati di una riga in base alle colonne
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <param name="row">Riga di dati</param>
      /// <returns>I dati splittati</returns>
      public static string[] SplitData(this TextLoader.Options options, string row)
      {
         var expr = @"(?<match>\w+)|\""(?<match>[\w\s";
         foreach (var separator in options.Separators != default && options.Separators.Length > 0 ? options.Separators : new[] { ',' })
            expr += new string(separator, 1);
         expr += @"]*)""";
         return Regex.Matches(row, expr).Cast<Match>().Select(m => m.Groups["match"].Value).ToArray();
      }
      /// <summary>
      /// Trasforma le opzioni di caricamento testi in schema di vista dati.
      /// </summary>
      /// <param name="textLoaderOptions">Opzioni di caricamento testi</param>
      /// <returns>Lo schema di vista dati</returns>
      public static DataViewSchema ToDataViewSchema(this TextLoader.Options textLoaderOptions) =>
         MachineLearningContext.Default.MLNET.Data.CreateTextLoader(textLoaderOptions).GetOutputSchema();
      #endregion
   }
}
