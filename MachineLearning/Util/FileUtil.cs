using System.IO;
using System.Threading.Tasks;

namespace MachineLearning.Util
{
   /// <summary>
   /// Utilita' per i files
   /// </summary>
   internal class FileUtil
   {
      /// <summary>
      /// Cancella un file con possibilita' di retry
      /// </summary>
      /// <param name="filePath">Path del file</param>
      /// <param name="retryCount">Numero di tentativi</param>
      /// <param name="retryDelayMs">Ritardo fra i tentativi</param>
      public static void Delete(string filePath, int retryCount = 20, int retryDelayMs = 500)
      {
         // Verifica esistenza del file
         if (!File.Exists(filePath))
            return;
         try {
            // Cancella il file
            File.Delete(filePath);
         }
         catch (IOException) {
            // Memorizza la data del file per controllare che non vi siano scritture successive
            var dateTime = File.GetLastWriteTimeUtc(filePath);
            // Contatore di tentativi
            var retry = 0;
            // Funzione di riprova
            void Retry()
            {
               // Verifica esistenza file
               if (!File.Exists(filePath))
                  return;
               try {
                  // Verifica se il file non e' cambiato e riprova a cancellarlo
                  if (File.GetLastWriteTimeUtc(filePath) == dateTime)
                     File.Delete(filePath);
               }
               catch {
                  // In caso di insuccesso riprova dopo il tempo specificato
                  if (++retry < retryCount)
                     Task.Delay(retryDelayMs).ContinueWith(t => Retry());
               }
            }
            // Lancia un nuovo tentativo dopo il tempo specificato
            if (++retry < retryCount)
               Task.Delay(retryDelayMs).ContinueWith(t => Retry());
            throw;
         }
      }
   }
}
